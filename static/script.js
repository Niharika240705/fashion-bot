function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function sendMessage() {
    const inputField = document.getElementById('user-input');
    const messageText = inputField.value.trim();

    if (messageText === "") return;

    // Add User Message
    addMessage(messageText, 'user-message');
    inputField.value = "";

    // Show typing indicator (optional, but nice)
    // For now, prompt response
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: messageText }),
    })
        .then(response => response.json())
        .then(data => {
            addMessage(data.response, 'bot-message');
        })
        .catch(error => {
            console.error('Error:', error);
            addMessage("Apologies, I am experiencing a temporary connection issue.", 'bot-message');
        });
}

function addMessage(text, className) {
    const chatBox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', className);

    const textP = document.createElement('p');
    textP.innerText = text;

    messageDiv.appendChild(textP);
    chatBox.appendChild(messageDiv);

    // Scroll to bottom
    chatBox.scrollTop = chatBox.scrollHeight;
}
