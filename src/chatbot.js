import React, { useState } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

const Chatbot = () => {
    const [messages, setMessages] = useState([]);
    const [userInput, setUserInput] = useState('');

    // For voice recognition
    const { transcript, resetTranscript } = useSpeechRecognition();

    const handleSendMessage = () => {
        if (userInput || transcript) {
            const message = userInput || transcript;
            setMessages([...messages, { text: message, sender: 'user' }]);
            generateBotResponse(message);
            setUserInput('');
        }
    };

    const generateBotResponse = (message) => {
        // Mock response for the bot
        const botMessage = `You asked: "${message}". Here is some information: ...`;
        setMessages([...messages, { text: botMessage, sender: 'bot' }]);
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
