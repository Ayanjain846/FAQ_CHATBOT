@echo off
echo Fixing NLTK punkt_tab resource...
cd /d "C:\Users\Ayan Jain\OneDrive\Desktop\CHATBOT"
call venv\Scripts\activate.bat
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')"
echo.
echo Download complete! You can now run: streamlit run app.py
pause