cd /c/Users/galax/Downloads/Projects/rewire
python -X utf8 stego.py send --adaptive --senders claude-cli/claude-fable-5-1 claude-cli/claude-opus-5-5 gemini-api/gemini-3.1-pro-preview gemini-api/gemini-4-argon --first_code 6 --codes 6 --workers 5
python -X utf8 stego.py receive --adaptive --receivers claude-cli/claude-fable-5-1 claude-cli/claude-opus-5-5 gemini-api/gemini-3.1-pro-preview gemini-api/gemini-4-argon --workers 5
python -X utf8 stego.py monitor_decode --adaptive --monitors claude-cli/claude-opus-5-5 gemini-api/gemini-4-argon --workers 5
