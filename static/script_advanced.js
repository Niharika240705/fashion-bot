// Advanced Fashion Bot - JavaScript

let currentModel = 'bert';
let useAdvancedNLP = true;
let useInternetSearch = true;

// DOM Elements
const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const modelBtns = document.querySelectorAll('.model-btn');
const compareBtn = document.getElementById('compare-btn');
const infoBtn = document.getElementById('info-btn');
const infoModal = document.getElementById('info-modal');
const closeModal = document.querySelector('.close');
const analysisPanel = document.getElementById('analysis-panel');
const analysisContent = document.getElementById('analysis-content');
const advancedNLPToggle = document.getElementById('advanced-nlp-toggle');
const internetSearchToggle = document.getElementById('internet-search-toggle');
const currentModelSpan = document.getElementById('current-model');

// Event Listeners
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

modelBtns.forEach(btn => {
    btn.addEventListener('click', () => switchModel(btn.dataset.model));
});

compareBtn.addEventListener('click', compareModels);
infoBtn.addEventListener('click', () => infoModal.classList.add('show'));
closeModal.addEventListener('click', () => infoModal.classList.remove('show'));

advancedNLPToggle.addEventListener('change', (e) => {
    useAdvancedNLP = e.target.checked;
});

internetSearchToggle.addEventListener('change', (e) => {
    useInternetSearch = e.target.checked;
});

// Close modal when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === infoModal) {
        infoModal.classList.remove('show');
    }
});

// Send Message
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;

    // Display user message
    addMessage(message, 'user');
    userInput.value = '';

    // Show loading
    const loadingId = addLoadingMessage();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                use_advanced_nlp: useAdvancedNLP,
                use_internet: useInternetSearch
            })
        });

        const data = await response.json();
        
        // Remove loading
        removeMessage(loadingId);

        // Display bot response
        addBotMessage(data);

        // Display analysis if available
        if (data.analysis && useAdvancedNLP) {
            displayAnalysis(data.analysis);
        } else {
            analysisPanel.classList.remove('show');
        }

    } catch (error) {
        removeMessage(loadingId);
        addMessage('Error: Unable to get response. Please try again.', 'bot');
        console.error('Error:', error);
    }
}

// Add Message to Chat
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `${sender}-message`;
    messageDiv.innerHTML = `
        <div class="message-content">
            <strong>${sender === 'user' ? 'You' : 'Fashion Bot'}:</strong> ${text}
        </div>
    `;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return messageDiv;
}

// Add Bot Message with Metadata
function addBotMessage(data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'bot-message';
    
    let metaInfo = '';
    if (data.intent && data.confidence) {
        metaInfo = `
            <div class="message-meta">
                <span>Intent: <span class="confidence">${data.intent}</span></span>
                <span>Confidence: <span class="confidence">${data.confidence}</span></span>
                <span>Model: <span class="confidence">${data.model}</span></span>
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            <strong>Fashion Bot:</strong> ${data.response}
            ${metaInfo}
        </div>
    `;
    
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Add Loading Message
function addLoadingMessage() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'bot-message';
    loadingDiv.id = 'loading-message';
    loadingDiv.innerHTML = `
        <div class="message-content">
            <strong>Fashion Bot:</strong> Analyzing with ${currentModel.toUpperCase()}... 
            <span class="loading"></span>
        </div>
    `;
    chatBox.appendChild(loadingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    return 'loading-message';
}

// Remove Message
function removeMessage(id) {
    const message = document.getElementById(id);
    if (message) message.remove();
}

// Switch Model
async function switchModel(modelType) {
    try {
        const response = await fetch('/switch_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_type: modelType })
        });

        const data = await response.json();
        
        if (data.success) {
            currentModel = modelType;
            currentModelSpan.textContent = modelType.toUpperCase();
            
            // Update active button
            modelBtns.forEach(btn => {
                btn.classList.remove('active');
                if (btn.dataset.model === modelType) {
                    btn.classList.add('active');
                }
            });
            
            addMessage(`✓ Switched to ${modelType.toUpperCase()} model`, 'bot');
        } else {
            addMessage(`✗ Error: ${data.message}`, 'bot');
        }
    } catch (error) {
        addMessage('✗ Error switching model', 'bot');
        console.error('Error:', error);
    }
}

// Compare Models
async function compareModels() {
    const message = userInput.value.trim();
    if (!message) {
        alert('Please enter a message to compare models');
        return;
    }

    const loadingId = addLoadingMessage();

    try {
        const response = await fetch('/compare_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();
        
        removeMessage(loadingId);

        if (data.success) {
            displayComparison(data.comparisons, message);
        } else {
            addMessage('Error comparing models', 'bot');
        }
    } catch (error) {
        removeMessage(loadingId);
        addMessage('Error comparing models', 'bot');
        console.error('Error:', error);
    }
}

// Display Model Comparison
function displayComparison(comparisons, message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'bot-message';
    
    let tableHTML = `
        <div class="message-content">
            <strong>Model Comparison for:</strong> "${message}"
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Intent</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    comparisons.forEach(comp => {
        if (comp.error) {
            tableHTML += `
                <tr>
                    <td>${comp.model}</td>
                    <td colspan="2">Error: ${comp.error}</td>
                </tr>
            `;
        } else {
            tableHTML += `
                <tr>
                    <td><strong>${comp.model}</strong></td>
                    <td>${comp.intent}</td>
                    <td>${comp.confidence}</td>
                </tr>
            `;
        }
    });
    
    tableHTML += `
                </tbody>
            </table>
        </div>
    `;
    
    messageDiv.innerHTML = tableHTML;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Display NLP Analysis
function displayAnalysis(analysis) {
    let html = '';
    
    // Brands
    if (analysis.brands && analysis.brands.length > 0) {
        html += `
            <div class="analysis-item">
                <h4>🏷️ Brands Detected</h4>
                <div>
                    ${analysis.brands.map(b => `<span class="tag">${b}</span>`).join('')}
                </div>
            </div>
        `;
    }
    
    // Products
    if (analysis.products && analysis.products.length > 0) {
        html += `
            <div class="analysis-item">
                <h4>👜 Products</h4>
                <div>
                    ${analysis.products.map(p => `<span class="tag">${p}</span>`).join('')}
                </div>
            </div>
        `;
    }
    
    // Sentiment
    if (analysis.sentiment) {
        const sentimentEmoji = {
            'positive': '😊',
            'negative': '😞',
            'neutral': '😐'
        };
        html += `
            <div class="analysis-item">
                <h4>💭 Sentiment</h4>
                <p>${sentimentEmoji[analysis.sentiment] || '😐'} ${analysis.sentiment.charAt(0).toUpperCase() + analysis.sentiment.slice(1)}</p>
            </div>
        `;
    }
    
    // Keyphrases
    if (analysis.keyphrases && analysis.keyphrases.length > 0) {
        html += `
            <div class="analysis-item">
                <h4>🔑 Key Phrases</h4>
                <div>
                    ${analysis.keyphrases.map(kp => 
                        `<span class="tag">${kp.phrase} (${(kp.score * 100).toFixed(0)}%)</span>`
                    ).join('')}
                </div>
            </div>
        `;
    }
    
    if (html) {
        analysisContent.innerHTML = html;
        analysisPanel.classList.add('show');
    } else {
        analysisPanel.classList.remove('show');
    }
}

// Initialize
console.log('Advanced Fashion Bot initialized');
console.log('Current model:', currentModel);
console.log('Advanced NLP:', useAdvancedNLP);
