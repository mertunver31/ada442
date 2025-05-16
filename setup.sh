#!/bin/bash

# Install python dependencies
pip install -r requirements.txt

# Create streamlit config directory if it doesn't exist
mkdir -p ~/.streamlit

# Write config file
cat > ~/.streamlit/config.toml << EOF
[theme]
primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"
EOF

# Write credentials file
cat > ~/.streamlit/credentials.toml << EOF
[general]
email = ""
EOF 