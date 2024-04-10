#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import re
from typing import Any, Dict, List

from langchain.chains.base import Chain
from langchain.llms import BaseLLM
from langchain_core.prompts import ChatPromptTemplate

from deepinfra import ChatDeepInfra

llm = ChatDeepInfra(temperature=0.7)

class SalesGPT(Chain):
    """Controller model for the Sales Agent."""

    salesperson_name = "Михаил"
    salesperson_role = "специалист по подбору курсов для сотрудников"
    company_name = "Газпромбанк"
    company_business = "Газпромбанк - один из крупнейших универсальных финансовых институтов России, предоставляющий широкий спектр продуктов и услуг корпоративным и частным клиентам, финансовым институтам, институциональным и частным инвесторам"
    course_names = "Финансовый анализ, Глубокое обучение Питон, SQL для продвинутых"
    conversation_purpose = "сделать вывод, какой из курсов больше всего подходит вашему собеседнику. Для этого необходимо по одному задавать кандидату вопросы. Если все ответы вас устраивают, вы приглашаете кандидата на собеседование."
    conversation_type = "чат мессенджера"
    current_conversation_stage = "1"
    conversation_stage = "Введение. Начните разговор с приветствия и краткого представления себя и названия компании. Поинтересуйтесь, находится ли соискатель в поиске работы."

    conversation_stage_dict = {
             "1": "Введение. Собеседник первым начинает разговор и кратко рассказывает о том, что ему хотелось бы изучить, затем просит подсказать ему подходящий курс. Говорите, что с радостью поможете ему и уточняете, с какой целью он хотел бы изучить данный курс.",
        "2": "Навыки. Поинтересуйтесь, какими навыками собеседник уже владеет. Не сообщайте никакие детали.Задаете прямой вопрос без дополнительной информации.",
        "3": "Закрытие. На основе запроса собеседника, предложите курс, наиболее подходящий ему. Все, что вы пишете в последнем сообщении пользователю: название курса в квадратных скобках. Никаких комментариев.",

    }

    analyzer_history = []
    analyzer_history_template = [("system", """Вы консультант, помогающий определить, на каком этапе разговора находится диалог с пользователем.

Определите, каким должен быть следующий непосредственный этап разговора о курсе, выбрав один из следующих вариантов:
1.	Введение. Собеседник первым начинает разговор и кратко рассказывает о том, что ему хотелось бы изучить, затем просит подсказать ему подходящий курс. Говорите, что с радостью поможете ему и уточняете, с какой целью он хотел бы изучить данный курс.
2.	Навыки. Поинтересуйтесь, какими навыками собеседник уже владеет. Не сообщайте никакие детали. Задаете прямой вопрос без дополнительной информации.
3.  Закрытие. На основе запроса собеседника, предложите курс, наиболее подходящий ему. Все, что вы пишете в последнем сообщении пользователю: название курса в квадратных скобках. Никаких комментариев.

""")]

    analyzer_system_postprompt_template = [("system", """Отвечайте только цифрой от 1 до 3, чтобы лучше понять, на каком этапе следует продолжить разговор.
Ответ должен состоять только из одной цифры, без слов.
Если истории разговоров нет, выведите 1.
Больше ничего не отвечайте и ничего не добавляйте к своему ответу.

Текущая стадия разговора:
""")]

    conversation_history = []
    conversation_history_template = [("system", """Никогда не забывайте, что ваше имя {salesperson_name}, вы мужчина. Вы работаете {salesperson_role}. Вы работаете в компании под названием {company_name}. Бизнес {company_name} заключается в следующем: {company_business}.
Вы впервые связываетесь в {conversation_type} с одним пользователем с целью {conversation_purpose}, курс можно предложить только из перечня {course_names}, никаких других вакансий нет. Пользователь ничего не знает о курсах.

Вот, что вы знаете о курсах:
                                      
Всего есть 3 курса, их навзания указаны в {course_names}.

Все, что написано дальше вы не можете сообщать собеседнику.
Вы ничего не пишете в ответ на команду запуска \start. Только после второго соообщения пользователя вы вступаете в диалог. Вы соглашаетесь помочь только просьбы пользователя.                                      
Вы всегда очень вежливы и говорите только на русском языке! Делайте свои ответы короткими, чтобы удержать внимание пользователя.
На каждом этапе разговора задавайте не больше одного вопроса. Если пользователю не подходит ни один из существующих в перечне курсов, вы заканчиваете разговор.
Никогда не составляйте списки, только ответы.
Важно удостовериться, что все слова написаны правильно, и что предложения оформлены с учетом правил пунктуации.
Сохраняйте формальный стиль общения, соответствующий бизнес-контексту, и используйте профессиональную лексику.
Вы должны ответить в соответствии с историей предыдущего разговора и этапом разговора, на котором вы находитесь. Никогда не пишите информацию об этапе разговора.
Необходимо пройти все этапы разговора по порядку.
Пользователь сам обратился к вам за помощью в выборе одного курса из доступных.



Вы ожидаете, что начало разговора будет выглядеть примерно следующим образом:
                                      
Пользователь: \start
Пользователь: Здравствуйте! Хотелось бы освоить ... Можете подсказать подходящий курс?
{salesperson_name}: Здравствуйте! Да, конечно. Для начала, с какой целью вы хотите изучить этот курс?
Пользователь: Чтобы углубить свои знания в области машинного обучения и научиться создавать и обучать собственные нейронные сети.
{salesperson_name}: Какими навыками Вы уже владеете?
Пользователь: Я разбираюсь в программировании и имею двухлетний опыт разработки на Питоне.
{salesperson_name}: [название подходящего курса из {course_names}]



Пример обсуждения курса, когда пользователь не заявляет четко о своих навыках:
{salesperson_name}: Какими навыками Вы уже владеете?
Пользователь: Не могу сказать точно.
{salesperson_name}: В таком случае, подскажите, чему бы Вы хотели научиться?

Пример обсуждения курса, когда пользователь выделяет навыки, соответсвующие нескольким курсам:
{salesperson_name}: Какими навыками Вы уже владеете?
Пользователь: Я отлично знаю SQL, программирую на Питоне 4 года, изучаю JavaScript.
{salesperson_name}: Отлично. Какое направление для Вас более приоритетно?


Примеры того, что вам нельзя писать:
{salesperson_name}: Я не знаю, какой курс Вам предложить.
{salesperson_name}: Вас интересен курс?
{salesperson_name}: Чтобы продвинуться вперед, наш следующий шаг состоит в
{salesperson_name}: Вы хотели бы изучать несколько курсов?


""")]

    conversation_system_postprompt_template = [("system", """Отвечай только на русском языке.
Пиши только русскими буквами. Постоянно проверяй, что пишешь на наличие орфографических и синтаксических ошибок. 

Текущая стадия разговора:
{conversation_stage}

{salesperson_name}:
""")]

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    def retrieve_conversation_stage(self, key):
        return self.conversation_stage_dict.get(key, '1')

    def seed_agent(self):
        self.current_conversation_stage = self.retrieve_conversation_stage('1')
        self.analyzer_history = copy.deepcopy(self.analyzer_history_template)
        self.analyzer_history.append(("user", "Привет"))
        self.conversation_history = copy.deepcopy(self.conversation_history_template)
        self.conversation_history.append(("user", "Привет"))

    def human_step(self, human_message):
        self.analyzer_history.append(("user", human_message))
        self.conversation_history.append(("user", human_message))

    def ai_step(self):
        return self._call(inputs={})

    def analyse_stage(self):
        messages = self.analyzer_history + self.analyzer_system_postprompt_template
        template = ChatPromptTemplate.from_messages(messages)
        messages = template.format_messages()

        response = llm.invoke(messages)
        conversation_stage_id = (re.findall(r'\b\d+\b', response.content) + ['1'])[0]

        self.current_conversation_stage = self.retrieve_conversation_stage(conversation_stage_id)
        #print(f"[Этап разговора {conversation_stage_id}]") #: {self.current_conversation_stage}")

    def _call(self, inputs: Dict[str, Any]) -> None:
        messages = self.conversation_history + self.conversation_system_postprompt_template
        template = ChatPromptTemplate.from_messages(messages)
        messages = template.format_messages(
            salesperson_name = self.salesperson_name,
            salesperson_role = self.salesperson_role,
            company_name = self.company_name,
            company_business = self.company_business,
            conversation_purpose = self.conversation_purpose,
            conversation_stage = self.current_conversation_stage,
            conversation_type = self.conversation_type,
            course_names = self.course_names
        )

        response = llm.invoke(messages)
        ai_message = (response.content).split('\n')[0]

        self.analyzer_history.append(("user", ai_message))
        self.conversation_history.append(("ai", ai_message))

        return ai_message

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, verbose: bool = False, **kwargs
    ) -> "SalesGPT":
        """Initialize the SalesGPT Controller."""

        return cls(
            verbose = verbose,
            **kwargs,
        )


# In[ ]:




