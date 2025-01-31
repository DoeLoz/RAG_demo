document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById('question-form');
    const answerText = document.getElementById('answer-text');

    form.addEventListener('submit', async function(event) {
        event.preventDefault();

        const question = document.getElementById('question-input').value.trim();

        if (!question) {
            answerText.innerHTML = "请输入问题！";
            return;
        }


        answerText.innerHTML = "正在处理中，请稍候...";

        try {

            const response = await fetch('/get_answer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question: question }),
            });

            const data = await response.json();
            if (data.answer) {
                answerText.innerHTML = `<strong>答案：</strong> ${data.answer}`;
            } else {
                answerText.innerHTML = "抱歉，无法回答这个问题。";
            }
        } catch (error) {
            answerText.innerHTML = "发生错误，请稍后重试！";
            console.error("API请求失败:", error);
        }
    });
});

