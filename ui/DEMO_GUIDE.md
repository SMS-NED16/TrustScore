# TrustScore UI - Demo Guide

## What You've Got

A complete, production-ready web UI for TrustScore that you can demo tomorrow!

### Features

✅ **Beautiful Modern UI** - Clean, professional design perfect for demos
✅ **Real-time Analysis** - Analyze LLM responses instantly
✅ **Visual Error Highlighting** - Errors are color-coded and highlighted in the response text
✅ **Category Breakdown** - See Trustworthiness, Explainability, and Bias scores separately
✅ **Confidence Intervals** - Statistical uncertainty displayed for all scores
✅ **Error Details** - Expandable cards showing error explanations and severity
✅ **Mock Mode** - Test without API calls for reliable demos
✅ **REST API** - Backend can also be used programmatically

## File Structure

```
ui/
├── app.py              # Flask backend server
├── static/
│   ├── index.html      # Main UI page
│   ├── styles.css      # Beautiful styling
│   └── app.js          # Frontend logic
├── requirements.txt    # UI dependencies (Flask, flask-cors)
├── README.md           # Full documentation
├── QUICKSTART.md       # Quick start guide
├── start.sh            # Linux/Mac startup script
└── start.bat           # Windows startup script
```

## Quick Start (For Demo)

### 1. Install (if not done)
```bash
cd ui
pip install -r requirements.txt
```

### 2. Run
```bash
python app.py
```

### 3. Open Browser
Go to: `http://localhost:5000`

## Demo Script

### Opening
1. Show the clean, modern interface
2. Explain: "This is TrustScore - a tool for evaluating LLM responses"

### Input Demo
1. Enter a prompt: "What is machine learning?"
2. Enter a response with errors: "Machine learning is AI. The capital of France is Paris, which is important for ML."
3. Check "Use Mock Mode" (for reliable demo)
4. Click "Analyze Response"

### Results Demo
1. **Overall Score**: Point out the large TrustScore number
2. **Category Breakdown**: Show T/E/B scores
3. **Error Highlighting**: Show how errors are highlighted in the response
4. **Error Details**: Click through error cards to show explanations

### Key Talking Points

- **TrustScore**: Lower is better (measures error severity)
- **Three Categories**: 
  - T (Trustworthiness): Factual errors, hallucinations
  - E (Explainability): Clarity, missing context
  - B (Bias): Demographic, gender, cultural bias
- **Confidence Intervals**: Show statistical uncertainty
- **Span-level Detection**: Errors are precisely located in text

## Troubleshooting

### Server Won't Start
- Check if port 5000 is in use
- Make sure Flask is installed: `pip install flask flask-cors`
- Check Python version (3.8+)

### Import Errors
- Make sure you're in the `ui` directory when running
- The app adds the parent directory to the path automatically

### Mock Mode Issues
- If mock mode doesn't work, try without it (if you have API key)
- Mock mode requires proper configuration in `config/settings.py`

### Static Files Not Loading
- Make sure `ui/static/` directory exists with all files
- Check browser console for 404 errors

## Tips for Great Demo

1. **Prepare Examples**: Have 2-3 example prompt/response pairs ready
2. **Use Mock Mode**: More reliable, no API rate limits
3. **Show Error Highlighting**: Use examples with clear, visible errors
4. **Explain Scores**: Be ready to explain what the numbers mean
5. **Show Confidence**: Point out confidence intervals to show rigor

## What Makes This Special

- **No Frontend Framework Needed**: Pure HTML/CSS/JS - easy to understand
- **Simple Backend**: Flask is straightforward and well-documented
- **Production Ready**: Error handling, loading states, responsive design
- **Extensible**: Easy to add features or modify

## Next Steps (After Demo)

If you want to enhance it:
- Add batch upload (CSV/JSON file)
- Add history/saved analyses
- Add export functionality (PDF reports)
- Add comparison view (side-by-side)
- Add user authentication

Good luck with your demo! 🚀

