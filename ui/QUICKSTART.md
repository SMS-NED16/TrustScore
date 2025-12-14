# TrustScore UI - Quick Start Guide

## For Tomorrow's Demo 🚀

### Step 1: Install Dependencies (One Time)

Open a terminal in the **root TrustScore directory** and run:

```bash
# Install main dependencies (if not already done)
pip install -r requirements.txt

# Install UI dependencies
cd ui
pip install -r requirements.txt
```

### Step 2: Start the Server

From the `ui` directory:

**Windows:**
```bash
python app.py
```

**Mac/Linux:**
```bash
python app.py
```

Or use the convenience scripts:
- Windows: `start.bat`
- Mac/Linux: `start.sh` (make executable first: `chmod +x start.sh`)

### Step 3: Open in Browser

The server will start on `http://localhost:5000`

Open your browser and navigate to that URL.

### Step 4: Demo Tips

1. **Use Mock Mode First**: Check the "Use Mock Mode" checkbox for reliable demos without API calls
2. **Try These Examples**:
   - **Good Response**: 
     - Prompt: "What is the capital of France?"
     - Response: "The capital of France is Paris."
   
   - **Response with Errors**:
     - Prompt: "Summarize the main points of machine learning."
     - Response: "Machine learning is a subset of AI that enables computers to learn. The capital of France is Paris, which is relevant to machine learning."

3. **Show Features**:
   - Overall TrustScore display
   - Category breakdown (T/E/B)
   - Error highlighting in response text
   - Error details with explanations

### Troubleshooting

**Port 5000 already in use?**
- Change the port in `app.py` (last line): `app.run(debug=True, host='0.0.0.0', port=8080)`

**Import errors?**
- Make sure you're running from the `ui` directory
- Make sure all dependencies are installed

**Mock mode not working?**
- This is expected if mock components aren't fully configured
- Try without mock mode if you have an API key

### What to Show in Demo

1. **Input Section**: Show how easy it is to input prompt/response
2. **Results Display**: 
   - Overall score (big number)
   - Category scores (T/E/B breakdown)
   - Visual error highlighting
3. **Error Details**: Click through error cards to show explanations
4. **Confidence Intervals**: Show statistical uncertainty

Good luck with your demo! 🎉








