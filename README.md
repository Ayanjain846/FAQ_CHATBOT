# 🤖 FAQ Chatbot - NLP-Powered Conversational AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)](https://streamlit.io)
[![NLTK](https://img.shields.io/badge/NLTK-3.8.1-green.svg)](https://nltk.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> *An intelligent FAQ chatbot that understands natural language queries using TF-IDF vectorization and cosine similarity*

## 📌 Table of Contents
- [Overview](#-overview)
- [Features](#-key-features)
- [Technical Architecture](#-technical-architecture)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Sample Interactions](#-sample-interactions)
- [NLP Pipeline Deep Dive](#-nlp-pipeline-deep-dive)
- [Future Enhancements](#-future-enhancements)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Author](#-author)

## 🎯 Overview

This project is a **production-ready FAQ chatbot** developed for the **CodeAlpha Internship Assignment**. It demonstrates the practical application of Natural Language Processing (NLP) techniques to create a conversational AI that understands user queries and provides relevant answers from a knowledge base.

Unlike simple keyword-matching bots, this system uses **semantic similarity** to understand the meaning behind user questions, making it robust to different phrasings and word choices.

## ✨ Key Features

### Core NLP Capabilities
- **Intelligent Question Matching**: Uses TF-IDF vectorization and cosine similarity to find semantically similar questions
- **Smart Text Preprocessing**: Implements comprehensive text cleaning (tokenization, stopword removal, punctuation filtering)
- **Keyword Boosting**: Enhances matching accuracy for critical topics (refunds, shipping, tracking)
- **Fallback Mechanisms**: Gracefully handles unmatched queries with helpful suggestions

### User Experience
- **Conversational Memory**: Maintains chat history using Streamlit's session state
- **Real-time Responses**: Instant feedback with loading indicators
- **Professional UI**: Clean, modern chat interface with side panel information
- **Confidence Indicators**: Shows match confidence for low-similarity responses

### Technical Robustness
- **Error Handling**: Graceful fallbacks when NLTK resources are missing
- **Threshold-based Retrieval**: Configurable similarity threshold (0.15 default)
- **Multi-lingual Ready**: Architecture supports easy language extension

## 🛠️ Technical Architecture

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend/UI** | Streamlit 1.28.1 | Web-based chat interface with real-time updates |
| **NLP Processing** | NLTK 3.8.1 | Text tokenization, stopword removal, preprocessing |
| **Vectorization** | scikit-learn 1.3.0 | TF-IDF transformation and cosine similarity |
| **Backend Logic** | Python 3.8+ | Core business logic and NLP pipeline |

