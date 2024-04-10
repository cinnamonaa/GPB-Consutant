#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import asyncio
import logging
import telebot

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram import types
from telebot import message_handler
from HR import SalesGPT, llm


bot_token = "7062390871:AAFePSVWcz9QTxPXbJRq7KP-BazQJsgYBgU"

sales_agent = None

async def main():

    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = Bot(bot_token, parse_mode=None)
    logging.basicConfig(level=logging.INFO)

    @dp.message(Command(commands=["start"]))
    async def repl(message):
        pass

    first_message_received = False

    @dp.message_handler(Message(message =  True))
    async def handle_text_message(message: types.Message):
        global first_message_received
    
        if not first_message_received:
            # Обрабатываем текстовое сообщение от пользователя
            print(message.text)
            
            # Устанавливаем флаг в True после обработки первого сообщения
            first_message_received = True



    @dp.message(F.text)
    async def repl(message):
        if sales_agent is None:
            await message.answer("Используйте команду /start")
        else:
            human_message = message.text
            if human_message:
                sales_agent.human_step(human_message)
                sales_agent.analyse_stage()
            ai_message = sales_agent.ai_step()
            await message.answer(ai_message)

    @dp.message(~F.text)
    async def empty(message):
        await message.answer("Бот принимает только текст")

    await dp.start_polling(bot)
    
    

if __name__ == "__main__":
    asyncio.run(main())

