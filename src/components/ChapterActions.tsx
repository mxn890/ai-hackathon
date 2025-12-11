import React, { useState } from 'react';
import styles from './ChapterActions.module.css';

interface ChapterActionsProps {
  content: string;
  onContentChange: (newContent: string) => void;
}

export default function ChapterActions({ content, onContentChange }: ChapterActionsProps) {
  const [loading, setLoading] = useState(false);
  const [isUrdu, setIsUrdu] = useState(false);
  const [originalContent, setOriginalContent] = useState(content);

  const personalizeContent = async () => {
    setLoading(true);
    try {
      // Get user profile from localStorage (we'll set this on login)
      const userProfile = JSON.parse(localStorage.getItem('userProfile') || '{}');
      
      const response = await fetch('http://localhost:8000/personalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: originalContent,
          userBackground: {
            experienceLevel: userProfile.experienceLevel || 'beginner',
            softwareBackground: userProfile.softwareBackground || '',
            hardwareBackground: userProfile.hardwareBackground || ''
          }
        })
      });

      const data = await response.json();
      onContentChange(data.personalizedContent);
      alert('Content personalized for your background! 🎯');
    } catch (error) {
      alert('Failed to personalize content. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const translateToUrdu = async () => {
    setLoading(true);
    try {
      const contentToTranslate = isUrdu ? originalContent : content;
      
      const response = await fetch('http://localhost:8000/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          content: contentToTranslate,
          targetLanguage: isUrdu ? 'english' : 'urdu'
        })
      });

      const data = await response.json();
      onContentChange(data.translatedContent);
      setIsUrdu(!isUrdu);
      alert(isUrdu ? 'Translated to English! 🇬🇧' : 'اردو میں ترجمہ کر دیا گیا! 🇵🇰');
    } catch (error) {
      alert('Failed to translate content. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetContent = () => {
    onContentChange(originalContent);
    setIsUrdu(false);
  };

  return (
    <div className={styles.chapterActions}>
      <div className={styles.actionsContainer}>
        <button 
          onClick={personalizeContent} 
          className={styles.actionButton}
          disabled={loading}
        >
          {loading ? '⏳ Processing...' : '🎯 Personalize Content'}
        </button>
        
        <button 
          onClick={translateToUrdu} 
          className={styles.actionButton}
          disabled={loading}
        >
          {loading ? '⏳ Processing...' : isUrdu ? '🇬🇧 Translate to English' : '🇵🇰 Translate to Urdu'}
        </button>

        <button 
          onClick={resetContent} 
          className={styles.resetButton}
          disabled={loading}
        >
          🔄 Reset
        </button>
      </div>
    </div>
  );
}