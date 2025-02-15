import React, { useState } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

const Chatbot = () => {
    const [messages, setMessages] = useState([]);
    const [userInput, setUserInput] = useState('');

    // For voice recognition
    const { transcript, resetTranscript } = useSpeechRecognition();

    const handleSendMessage = async () => {
        if (userInput || transcript) {
            const message = userInput || transcript;
            setMessages([...messages, { text: message, sender: 'user' }]);

            // Make API call to Flask backend
            const response = await fetch('http://127.0.0.1:5000/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message }),
            });
            const data = await response.json();
            const botMessage = data.response;

            // Update bot's response in the chat
            setMessages([...messages, { text: botMessage, sender: 'bot' }]);
            setUserInput(''); // Clear input field
        }
    };

    const handleVoiceRecognition = () => {
        SpeechRecognition.startListening({ continuous: true, language: 'en-US' });
    };

    const handleStopVoiceRecognition = () => {
        SpeechRecognition.stopListening();
    };

    const handleDownloadDocument = (docName) => {
        const fileUrl = `/path/to/your/documents/${docName}.pdf`;
        const link = document.createElement('a');
        link.href = fileUrl;
        link.download = `${docName}.pdf`;
        link.click();
    };

    return (
        <div className="chatbot-container">
            <div className="chatbox">
                <div className="chat-header">
                    <h3>Healthcare Chatbot</h3>
                </div>

                <div className="chat-window">
                    {messages.map((message, index) => (
                        <div
                            key={index}
                            className={message.sender === 'user' ? 'user-message' : 'bot-message'}
                        >
                            <p>{message.text}</p>
                        </div>
                    ))}
                </div>

                <div className="chat-footer">
                    <input
                        type="text"
                        value={userInput}
                        onChange={(e) => setUserInput(e.target.value)}
                        placeholder="Ask a question..."
                    />
                    <button onClick={handleSendMessage}>Send</button>
                    <button onClick={handleVoiceRecognition}>Start Voice</button>
                    <button onClick={handleStopVoiceRecognition}>Stop Voice</button>
                </div>

                {/* Button for document download */}
                <div className="download-docs">
                    <button onClick={() => handleDownloadDocument('healthcare-guide')}>
                        Download Healthcare Guide
                    </button>
                </div>
            </div>
        </div>
    );
};

export default Chatbot;
