# pip install zhipuai

from zhipuai import ZhipuAI

client = ZhipuAI(api_key="yourkey") #put your key here

response = client.chat.completions.create(
    model="glm-4-air-0111",
    messages=[
        {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
        {"role": "user", "content": "your problem"} #enter your problem here and see the result
    ],
)
print(response.choices[0].message)


