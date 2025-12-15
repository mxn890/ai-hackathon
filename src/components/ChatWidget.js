import React, { useState, useRef, useEffect } from 'react';
import styles from './ChatWidget.module.css';

export default function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [userProfile, setUserProfile] = useState(null);
  const messagesEndRef = useRef(null);

  // Initialize user profile on mount
  useEffect(() => {
    // Check if user profile exists in localStorage
    const storedProfile = localStorage.getItem('userProfile');
    if (storedProfile) {
      setUserProfile(JSON.parse(storedProfile));
    } else {
      // Set a default profile for demo purposes
      const defaultProfile = {
        id: 'demo_user_' + Date.now(),
        name: 'Student',
        experienceLevel: 'beginner',
        softwareBackground: 'Basic Python',
        hardwareBackground: 'None',
        topicsInterest: ['robotics', 'ai', 'physical-ai'],
        lastActive: new Date().toISOString()
      };
      localStorage.setItem('userProfile', JSON.stringify(defaultProfile));
      setUserProfile(defaultProfile);
    }
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Detect text selection
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      const text = selection.toString().trim();
      if (text && text.length > 10) {
        setSelectedText(text);
        
        // Show notification that text has been captured
        const notification = document.createElement('div');
        notification.className = styles.selectionNotification;
        notification.innerHTML = `
          <div style="
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            z-index: 10000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
          ">
            ✅ Text captured! You can now ask questions about it.
          </div>
        `;
        document.body.appendChild(notification);
        setTimeout(() => {
          if (document.body.contains(notification)) {
            document.body.removeChild(notification);
          }
        }, 3000);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => document.removeEventListener('mouseup', handleSelection);
  }, []);

  const clearSelectedText = () => {
    setSelectedText('');
    window.getSelection().removeAllRanges(); // Clear actual selection
  };

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { 
      role: 'user', 
      content: input,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Prepare request data with user context
      const requestData = {
        message: input,
        selected_text: selectedText || null,
        user_profile: userProfile || null,
        chat_history: messages.slice(-5).map(m => ({
          role: m.role,
          content: m.content
        }))
      };

      const response = await fetch('https://fast-api-chatbot1-3.onrender.com/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      const botMessage = { 
        role: 'assistant', 
        content: data.response,
        sources: data.sources || [],
        timestamp: new Date().toISOString(),
        response_time: data.response_time || null
      };
      
      setMessages(prev => [...prev, botMessage]);
      clearSelectedText();
      
      // Update user profile based on interaction
      if (userProfile) {
        const updatedProfile = {
          ...userProfile,
          lastActive: new Date().toISOString(),
          totalQueries: (userProfile.totalQueries || 0) + 1
        };
        localStorage.setItem('userProfile', JSON.stringify(updatedProfile));
        setUserProfile(updatedProfile);
      }
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage = { 
        role: 'assistant', 
        content: `❌ Error: ${error.message}. Please try again or check if the backend server is running.`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setSelectedText('');
  };

  const handleSuggestionClick = (suggestion) => {
    setInput(suggestion);
    // Auto-send after short delay
    setTimeout(() => {
      sendMessage();
    }, 100);
  };

  // Suggested questions for beginners
  const suggestions = [
    "What is Physical AI?",
    "Explain bipedal locomotion",
    "How do humanoid robots perceive their environment?",
    "What are the main challenges in humanoid robotics?"
  ];

  return (
    <>
      {/* Floating Chat Button */}
      <button 
        className={styles.chatButton}
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle chat"
        data-open={isOpen}
      >
        {isOpen ? '✕' : '💬'}
        {messages.length > 0 && !isOpen && (
          <span className={styles.notificationBadge}>{messages.length}</span>
        )}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className={styles.chatWindow}>
          <div className={styles.chatHeader}>
            <div className={styles.headerContent}>
              <h3>🤖 Physical AI Assistant</h3>
              <div className={styles.headerActions}>
                {selectedText && (
                  <button 
                    onClick={clearSelectedText}
                    className={styles.clearSelectionBtn}
                    title="Clear selected text"
                  >
                    📝 Clear Selection
                  </button>
                )}
                <button 
                  onClick={clearChat}
                  className={styles.clearChatBtn}
                  title="Clear chat history"
                >
                  🗑️ Clear
                </button>
              </div>
            </div>
            
            <div className={styles.userInfo}>
              <span>👤 {userProfile?.name || 'Student'}</span>
              <span>📊 Level: {userProfile?.experienceLevel || 'beginner'}</span>
            </div>
          </div>

          <div className={styles.chatMessages}>
            {messages.length === 0 && (
              <div className={styles.welcomeMessage}>
                <div className={styles.welcomeHeader}>
                  <h4>👋 Welcome to Physical AI Assistant!</h4>
                  <p>I'm here to help you learn about Physical AI & Humanoid Robotics.</p>
                </div>
                
                <div className={styles.quickTips}>
                  <p><strong>💡 Quick Tips:</strong></p>
                  <ul>
                    <li>Select text on the page and ask questions about it</li>
                    <li>Ask follow-up questions for deeper understanding</li>
                    <li>Click suggestions below for quick start</li>
                  </ul>
                </div>
                
                <div className={styles.suggestions}>
                  <p><strong>🎯 Try asking:</strong></p>
                  <div className={styles.suggestionButtons}>
                    {suggestions.map((suggestion, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleSuggestionClick(suggestion)}
                        className={styles.suggestionBtn}
                      >
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}
            
            {messages.map((msg, idx) => (
              <div 
                key={idx} 
                className={`${styles.messageContainer} ${
                  msg.role === 'user' ? styles.userContainer : styles.botContainer
                }`}
              >
                <div className={styles.messageMeta}>
                  <span className={styles.messageRole}>
                    {msg.role === 'user' ? '👤 You' : '🤖 Assistant'}
                  </span>
                  <span className={styles.messageTime}>
                    {msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString([], { 
                      hour: '2-digit', 
                      minute: '2-digit' 
                    }) : ''}
                  </span>
                </div>
                
                <div className={
                  msg.role === 'user' ? styles.userMessage : styles.botMessage
                }>
                  <div className={styles.messageContent}>
                    {msg.content}
                  </div>
                  
                  {msg.sources && msg.sources.length > 0 && (
                    <div className={styles.sources}>
                      <div className={styles.sourcesHeader}>
                        📚 <strong>Reference Sources:</strong>
                      </div>
                      {msg.sources.map((source, i) => (
                        <div key={i} className={styles.source}>
                          <a 
                            href={`#${source.source}`} 
                            target="_blank" 
                            rel="noopener noreferrer"
                          >
                            📄 {source.source}
                          </a>
                          {source.page && <span className={styles.pageRef}>Page {source.page}</span>}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {loading && (
              <div className={styles.botContainer}>
                <div className={styles.messageMeta}>
                  <span className={styles.messageRole}>🤖 Assistant</span>
                  <span className={styles.messageTime}>Just now</span>
                </div>
                <div className={styles.botMessage}>
                  <div className={styles.typingIndicator}>
                    <div className={styles.typing}>
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                    <span className={styles.typingText}>Thinking...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          <div className={styles.chatInputContainer}>
            {selectedText && (
              <div className={styles.selectedTextPreview}>
                <div className={styles.selectedTextHeader}>
                  <strong>📝 Selected Text:</strong>
                  <button 
                    onClick={clearSelectedText}
                    className={styles.clearSelectionSmall}
                    title="Clear selection"
                  >
                    ✕
                  </button>
                </div>
                <div className={styles.selectedTextContent}>
                  {selectedText.length > 100 
                    ? selectedText.substring(0, 100) + '...' 
                    : selectedText}
                </div>
              </div>
            )}
            
            <div className={styles.chatInput}>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a question about Physical AI or Humanoid Robotics..."
                rows="2"
                disabled={loading}
              />
              <button 
                onClick={sendMessage}
                disabled={loading || !input.trim()}
                className={styles.sendButton}
                title="Send message"
              >
                {loading ? (
                  <span className={styles.loadingSpinner}></span>
                ) : (
                  '➤'
                )}
              </button>
            </div>
            
            <div className={styles.inputFooter}>
              <span className={styles.hint}>
                Press <kbd>Enter</kbd> to send, <kbd>Shift + Enter</kbd> for new line
              </span>
              <span className={styles.messageCount}>
                Messages: {messages.length}
              </span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}