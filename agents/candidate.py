import json
from http.client import responses

from langchain.agents import create_agent

from models.user import DingdingUserModel
from repository.interview_repo import InterviewRepo
from schemas.candidate_schema import CandidateSchema
from schemas.position_schema import PositionSchema
from schemas.user_schema import UserSchema
from langgraph.graph.message import BaseMessage

from utils.available_time import find_available_slot
from utils.iso8601 import iso8601_to_datetime_beijing
from .llms import qwen_llm,deepseek_llm
from .prompts import CANDIDATE_PROCESS_SYSTEM_PROMPT,SCORE_FOR_CANDIDATE_SYSTEM_PROMPT,SCORE_FOR_CANDIDATE_USER_PROMPT
from langchain.agents.middleware import ModelFallbackMiddleware,SummarizationMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from settings import settings
from pydantic import BaseModel
from typing import Annotated, List, TypeVar, Optional
from langgraph.graph.message import add_messages
from langchain.tools import tool,ToolRuntime
from schemas.agent_schema import AgentCandidateScoreSchema
from langchain_core.prompts import PromptTemplate
from models import AsyncSession,AsyncSessionFactory
from repository.candidate_repo import CandidateAIScoreRepo, CandidateRepo
from models.candidate import CandidateStatusEnum
from loguru import logger
from repository.user_repo import UserRepo
from core.dingtalk import DingTalkHttp
from core.cache import HRCache
from core.cache import DingTalkTokenInfoSchema
from datetime import datetime,timedelta
from typing import Any
from core.email_bot import EmailBot,EmailBotSettings
from models.interview import InterviewResultEnum

async def get_dingtalk_access_token(user_id: str) -> str:
    dingding_http = DingTalkHttp()

    # 2. 从缓存中获取该用户的refresh_token
    cache: HRCache = HRCache()
    token_info = await cache.get_dingtalk_info(user_id)
    if not token_info:
        error_message = f"{user_id}用户钉钉授权已过期！"
        logger.error(error_message)
        raise ValueError(error_message)

    try:
        # 3. 根据refresh_token刷新access_token
        refresh_token, access_token = await dingding_http.refresh_access_token(token_info.refresh_token)

        # 4. 将获取到的token信息重新设置到缓存中
        await cache.set_dingtalk_info(
            DingTalkTokenInfoSchema(
                user_id=user_id,
                access_token=access_token,
                refresh_token=refresh_token
            )
        )

        return access_token
    except Exception as e:
        logger.error(e)
        raise ValueError(e)


T = TypeVar("T")
#数据处理方法：当有新值（right）时更新状态，否则保持原值（left）
def assign_state_property(left: T, right: Optional[T]) -> T:
    return right if right is not None else left

# checkpointer会自动保存state中的数据到数据库中
class CandidateAgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    candidate: Annotated[CandidateSchema, assign_state_property]
    position: Annotated[PositionSchema, assign_state_property]
    interviewer: Annotated[UserSchema, assign_state_property]
@tool
async def score_for_candidate(
    runtime: ToolRuntime[CandidateAgentState]
):
    """
    根据职位信息，给职位上的候选人进行评分。
    评分结果会存入数据库中，并根据评分结果修改候选人状态。
    return：
    - str | None: 评分结果的JSON字符串，如果评分失败则返回None
    """
    candidate: CandidateSchema = runtime.state['candidate']
    position: PositionSchema = runtime.state['position']

    score_agent = create_agent(
        model=qwen_llm,
        system_prompt=SCORE_FOR_CANDIDATE_SYSTEM_PROMPT,
        middleware=[ModelFallbackMiddleware(first_model=deepseek_llm)],
        response_format=AgentCandidateScoreSchema
    )
    user_prompt_template = PromptTemplate.from_template(SCORE_FOR_CANDIDATE_USER_PROMPT)
    user_prompt = user_prompt_template.invoke({
        "candidate": candidate.model_dump_json(),
        "position": position.model_dump_json()
    })
    response = await score_agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": user_prompt.text
        }]
    })
    candiate_score: AgentCandidateScoreSchema = response['structured_response']
    # 将得分情况存储到数据库中
    async with AsyncSessionFactory() as session:
        async with session.begin():
            try:
                score_repo = CandidateAIScoreRepo(session)
                candidate_repo = CandidateRepo(session)
                # 1. 将得分插入到数据库中
                await score_repo.create_candidate_score(
                    candidate_id=candidate.id,
                    candidate_score_dict=candiate_score.model_dump()
                )
                # 2. 判断得分情况，如果超过8分，那么修改状态为AI_PASS，否则就是AI_FAILED
                status = CandidateStatusEnum.AI_FILTER_FAILED
                if candiate_score.overall_score > 8:
                    status = CandidateStatusEnum.AI_FILTER_PASSED

                await candidate_repo.update_candidate_status(candidate_id=candidate.id, status=status)
            except Exception as e:
                logger.error(e)
                raise ValueError(e)
                return f"得分工具执行失败，错误信息为：{e}"
    return f"得分工具执行成功！该候选人得分为：{candiate_score.model_dump_json()}"

@tool
async def get_interviewer_available_slot(
    runtime: ToolRuntime[CandidateAgentState]
):
    """
    根据职位信息，获取面试官可用的面试时间。
    """
    interviewer: UserSchema = runtime.state['interviewer']
    # 1. 获取该用户的钉钉账号
    union_id: str|None = None
    async with AsyncSessionFactory() as session:
        async with session.begin():
            try:
                user_repo = UserRepo(session)
                dingding_user: DingdingUserModel = await user_repo.get_dingding_user(user_id=interviewer.id)
                if not dingding_user:
                    return f"获取候选人可用时间失败，没有绑定钉钉账号！"
                union_id = dingding_user.union_id
            except Exception as e:
                return f"获取候选人可用时间失败：{e}"

    try:
        # 2. 获取access_token
        access_token: str = await get_dingtalk_access_token(interviewer.user_id)
    except Exception as e:
        logger.error(e)
        return f"获取面试官可用时间失败：{e}"

    # 3. 从钉钉上获取面试官的日程安排
    try:
        dingtalk_http = DingTalkHttp()
        now = datetime.now()
        tomorrow_nine = datetime(year=now.year, month=now.month, day=now.day, hour=9)
        events: list[dict[str, Any]] = await dingtalk_http.get_calendar_list(
            union_id=union_id,
            access_token=access_token,
            time_min=tomorrow_nine,
            time_max=tomorrow_nine + timedelta(days=7),#获取近七天内的空闲时间
        )
        #将钉钉API返回的ISO8601格式时间字符串转换为datetime对象
        busy_slots = [
            (iso8601_to_datetime_beijing(event['start']['dateTime']), iso8601_to_datetime_beijing(event['end']['dateTime']))
            for event in events
        ]#繁忙时间
        available_slots: list[tuple[datetime, datetime]] = find_available_slot(
            busy_slots,
            start_date=tomorrow_nine
        )
        if len(available_slots) == 0:
            return f"获取面试官可用时间失败：7天内没有空闲时间！"
        #将datatime对象转换为ISO8601时间格式
        available_times = [(iso8601_to_datetime_beijing(slot[0]), iso8601_to_datetime_beijing(slot[1])) for slot in available_slots]
        #返回json格式的字符串给大模型看
        return f"找到面试官可用的时间：{json.dumps(available_times)}"
    except Exception as e:
        return f"获取面试官可用时间失败：{e}"

@tool
async def send_interview_email(
    interview_datetime_str: str,
    runtime: ToolRuntime[CandidateAgentState],
):
    """
    给候选人发送面试时间邀请（并非最终面试时间，后续可能还需要通过邮件来和候选人协商最终面试时间）
    :param interview_datetime_str: 面试时间的字符串
    """
    candidate = runtime.state['candidate']
    position = runtime.state['position']

    email_bot_settings = EmailBotSettings(
        imap_host=settings.EMAIL_BOT_IMAP_HOST,
        smtp_host=settings.EMAIL_BOT_SMTP_HOST,
        email=settings.EMAIL_BOT_EMAIL,
        password=settings.EMAIL_BOT_PASSWORD,
    )
    async with EmailBot(email_bot_settings) as bot:
        subject = "面试邀请-协商面试时间"
        body = f"""
尊敬的{candidate.name}，
您好！
感谢您投递我司{position.title}职位。
我们初步确定了您的面试时间，请您确认是否方便。
面试时间：{interview_datetime_str}
请您确认是否方便，如果方便，请您回复“确认”。
如果不方便，请回复您方便的时间，我们将会重新协商面试时间。
谢谢！
        """
        try:
            await bot.send_email(
                to=candidate.email,
                subject=subject,
                text=body,
            )
        except Exception as e:
            logger.error(e)
            return f"给候选人发送邮件失败：{e}"

        return f"给候选人发送面试邀请邮件成功！面试时间初步确定为：{interview_datetime_str}"

@tool
async def confirm_interview_time(
    interview_datetime_str: str,#时间
    runtime: ToolRuntime[CandidateAgentState],
):
    """
    确认最终的面试时间。这个工具会做以下四件事：
    * 通过邮件，发送最终确认面试的时间给候选人
    * 给面试官的钉钉创建一个面试的日程安排
    * 在系统中创建一个面试预约记录
    # 在系统中修改候选人的状态为待面试
    :param interview_datetime_str: 面试时间，ISO8601格式的字符串
    """
    position = runtime.state['position']
    candidate = runtime.state['candidate']
    interviewer = runtime.state['interviewer']

    try:
        interview_datetime: datetime = iso8601_to_datetime_beijing(interview_datetime_str)
        # 由于后面存储数据库，不能在日期中带时区，所以这里先处理一下
        if interview_datetime.tzinfo is not None:
            interview_datetime_without_tz = interview_datetime.astimezone(None).replace(tzinfo=None)
    except Exception as e:
        return f"{interview_datetime_str}格式化失败！{e}"

    # 1. 通过邮件，发送最终确认面试的时间给候选人
    email_bot_settings = EmailBotSettings(
        imap_host=settings.EMAIL_BOT_IMAP_HOST,
        smtp_host=settings.EMAIL_BOT_SMTP_HOST,
        email=settings.EMAIL_BOT_EMAIL,
        password=settings.EMAIL_BOT_PASSWORD,
    )
    async with EmailBot(email_bot_settings) as bot:
        subject = "面试时间确定"
        body = f"""
        "尊敬的{candidate.name}，
        面试时间已确定：
        {interview_datetime_str}
        请您准时参加面试。该邮件无需再回复。谢谢！
        """
        try:
            await bot.send_email(
                to=candidate.email,
                subject=subject,
                text=body,
            )
        except Exception as e:
            logger.error(e)
            return f"给候选人发送邮件失败：{e}"

    # 2. 给面试官的钉钉创建一个面试的日程安排
    union_id: str|None = None
    try:
        async with AsyncSessionFactory() as session:
            async with session.begin():
                user_repo = UserRepo(session)
                dingding_user = await user_repo.get_dingding_user(user_id=interviewer.id)
                if not dingding_user:
                    return "面试官没有绑定钉钉账号！无法创建日程安排！"
                union_id = dingding_user.union_id
    except Exception as e:
        return f"面试官用户信息获取失败！"

    try:
        access_token: str = await get_dingtalk_access_token(interviewer.id)
    except Exception as e:
        return f"面试官access_token获取失败！{e}"

    try:
        dingtalk_http = DingTalkHttp()
        end_datetime: datetime = interview_datetime + timedelta(hours=1)
        await dingtalk_http.create_calendar(
            union_id=union_id,
            access_token=access_token,
            summary=f"面试安排：{position.title} - {candidate.name}",
            start_datetime=interview_datetime,
            end_datetime=end_datetime,
        )
    except Exception as e:
        return f"给面试官创建钉钉日程安排失败！{e}"

    try:
        async with AsyncSessionFactory() as session:
            async with session.begin():
                interview_repo = InterviewRepo(session)
                # 3. 在数据库中创建一个面试预约记录
                await interview_repo.create_interview({
                    # 面试时间因为是存储到postgresql数据库中，现在是只能存储没有带时区的日期
                    "scheduled_time": interview_datetime_without_tz,
                    "result": InterviewResultEnum.PENDING,
                    "candidate_id": candidate.id,
                    "interviewer_id": interviewer.id,
                })
                # 4. 在数据库中修改候选人的状态为待面试
                candidate_repo = CandidateRepo(session)
                await candidate_repo.update_candidate_status(
                    candidate_id=candidate.id,
                    status=CandidateStatusEnum.WAITING_FOR_INTERVIEW,
                )
    except Exception as e:
        return f"在系统中创建面试预约记录和候选人状态修改失败！{e}"

    return f"""
    * 给候选人发送面试时间执行成功！
    * 给面试官创建钉钉日程安排成功！
    * 在系统中创建面试预约成功！
    * 在系统中修改候选人状态为待面试成功！
    """
@tool
async def refuse_interview(
    runtime: ToolRuntime[CandidateAgentState],
):
    """
    如果候选人拒绝了面试，那么调用该工具来更新候选人状态为拒绝面试
    """
    candidate = runtime.state['candidate']
    try:
        async with AsyncSessionFactory() as session:
            async with session.begin():
                candidate_repo = CandidateRepo(session)
                await candidate_repo.update_candidate_status(
                    candidate_id=candidate.id,
                    status=CandidateStatusEnum.REFUSED_INTERVIEW,
                )
        return "已修改候选人状态为拒绝面试！"
    except Exception as e:
        return f"修改候选人状态为拒绝面试失败！"
@tool
def get_current_time() -> str:
    """
    获取当前时间（北京时间）：年月日、时分秒、星期几、当月第几天
    """
    now_bj = datetime.now()

    weekday_map = {
        0: "星期一",
        1: "星期二",
        2: "星期三",
        3: "星期四",
        4: "星期五",
        5: "星期六",
        6: "星期日",
    }
    weekday_cn = weekday_map[now_bj.weekday()]

    day_of_month = now_bj.day

    return (
        f"{now_bj.year}年{now_bj.month}月{now_bj.day}日 "
        f"{now_bj.hour:02d}:{now_bj.minute:02d}:{now_bj.second:02d} "
        f"{weekday_cn}（本月第{day_of_month}天）"
    )

class CandidateProcessAgent:
    def __init__(self,
            candidate: CandidateSchema|None=None,
            position: PositionSchema|None=None,
            interviewer: UserSchema|None=None
):
        self.candidate = candidate
        self.position = position
        self.interviewer = interviewer
        self._checkpointer=None

    async def ainvoke(self, messages: list[BaseMessage], thread_id: str):
        assert self._checkpointer is not None
        agent = create_agent(
            model=deepseek_llm,
            system_prompt=CANDIDATE_PROCESS_SYSTEM_PROMPT,
            state_schema=CandidateAgentState,
            middleware=[
                ModelFallbackMiddleware(first_model=deepseek_llm),
                SummarizationMiddleware(
                    model=qwen_llm,
                    trigger=("tokens", 50000),
                    keep=("tokens", 10000)
                )
            ],
            tools=[score_for_candidate,
                   get_interviewer_available_slot,
                   send_interview_email,
                   confirm_interview_time,
                   refuse_interview,
                   get_current_time,
            ],
            checkpointer=self._checkpointer
        )
        response = await agent.ainvoke({
            "messages": messages,
            "candidate": self.candidate,
            "position": self.position,
            "interviewer": self.interviewer,
        }, {"thread_id": thread_id})
        return response

    #进入上下文时执行
    async def __aenter__(self):
        # langchain如果大模型选择了一个工具，那么这个工具消息后的消息必须是工具调用后的结果，否则会报错
        self._checkpointer_conn = AsyncPostgresSaver.from_conn_string(settings.DATABASE_AGENT_URL)
        self._checkpointer = await self._checkpointer_conn.__aenter__()#建立数据库链接
        await self._checkpointer.setup()#创建表
        #这其实是两个async with ... as ..: 的嵌套但是因为嵌套了，所以
        # 嵌套在里面的若结束则会自动关闭无法走完流程，__aenter__()，__aexit__（）
        #这两个方法代表被嵌套在里面的async with的手动开始和手动结束
        # 等价于
        # async with AsyncPostgresSaver.from_conn_string(url) as cp:
        #     # ↑ Python 自动调了 .__aenter__() ← 建立连接
        #     await cp.setup()
        #     # 业务代码...
        # # ↑ Python 自动调了 .__aexit__() ← 关闭连接

        return self
    #退出上下文时执行
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._checkpointer_conn.__aexit__(exc_type, exc_val, exc_tb)
