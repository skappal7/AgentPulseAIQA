"""
Authentication Module
Handles user authentication with glassmorphic UI
"""

import streamlit as st
import hashlib
from typing import Optional, Dict


def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(password) == hashed


def load_glassmorphic_css(background_url: Optional[str] = None):
    """Load glassmorphic CSS styling for login page"""
    
    # Default gradient background if no image provided
    background_style = f"""
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    """ if not background_url else f"""
        background: url('{background_url}') center/cover no-repeat;
        background-attachment: fixed;
    """
    
    css = f"""
    <style>
        /* Hide Streamlit default elements */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* Full screen background */
        .stApp {{
            {background_style}
        }}
        
        /* Glassmorphic container */
        .glass-container {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            padding: 40px;
            max-width: 450px;
            margin: 100px auto;
        }}
        
        /* Login form styling */
        .login-title {{
            color: white;
            text-align: center;
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        .login-subtitle {{
            color: rgba(255, 255, 255, 0.9);
            text-align: center;
            font-size: 1.1em;
            margin-bottom: 30px;
        }}
        
        /* Input fields */
        .stTextInput > div > div > input {{
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            color: white;
            font-size: 16px;
            padding: 12px;
        }}
        
        .stTextInput > div > div > input::placeholder {{
            color: rgba(255, 255, 255, 0.7);
        }}
        
        .stTextInput > div > div > input:focus {{
            border-color: rgba(255, 255, 255, 0.5);
            box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.2);
        }}
        
        /* Button styling */
        .stButton > button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 40px;
            font-size: 16px;
            font-weight: 600;
            width: 100%;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }}
        
        /* Error message */
        .stAlert {{
            background: rgba(255, 68, 68, 0.2);
            border: 1px solid rgba(255, 68, 68, 0.4);
            border-radius: 10px;
            color: white;
        }}
        
        /* Labels */
        .stTextInput > label {{
            color: white !important;
            font-weight: 500;
        }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


def login_page(background_url: Optional[str] = None) -> bool:
    """
    Display glassmorphic login page
    
    Args:
        background_url: Optional URL to background image
        
    Returns:
        True if login successful, False otherwise
    """
    
    # Apply glassmorphic styling
    load_glassmorphic_css(background_url)
    
    # Center container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        
        # Title
        st.markdown('<h1 class="login-title">🚀 AgentPulse AI</h1>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">Enterprise QA & Coaching Platform</p>', unsafe_allow_html=True)
        
        # Login form
        username = st.text_input("Username", key="login_username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
        
        # Remember me
        remember_me = st.checkbox("Remember me", key="remember_me")
        
        # Login button
        login_button = st.button("Login", key="login_button", use_container_width=True)
        
        if login_button:
            if authenticate_user(username, password):
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.session_state['remember_me'] = remember_me
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return st.session_state.get('authenticated', False)


def authenticate_user(username: str, password: str) -> bool:
    """
    Authenticate user against Streamlit secrets
    
    Args:
        username: Username
        password: Password
        
    Returns:
        True if authenticated, False otherwise
    """
    
    try:
        # Get credentials from Streamlit secrets
        if hasattr(st, 'secrets') and 'passwords' in st.secrets:
            stored_hash = st.secrets['passwords'].get(username)
            
            if stored_hash:
                # Check if password matches
                return verify_password(password, stored_hash)
        
        # Fallback: hardcoded demo credentials (REMOVE IN PRODUCTION)
        demo_users = {
            'admin': hash_password('admin123'),
            'demo': hash_password('demo123')
        }
        
        if username in demo_users:
            return verify_password(password, demo_users[username])
        
        return False
        
    except Exception as e:
        st.error(f"Authentication error: {str(e)}")
        return False


def logout():
    """Logout current user"""
    st.session_state['authenticated'] = False
    st.session_state['username'] = None
    st.rerun()


def require_authentication(background_url: Optional[str] = None) -> bool:
    """
    Decorator-style function to require authentication
    
    Args:
        background_url: Optional URL to background image
        
    Returns:
        True if authenticated, shows login page otherwise
    """
    
    # Check if user is already authenticated
    if st.session_state.get('authenticated', False):
        return True
    
    # Show login page
    return login_page(background_url)


def get_current_user() -> Optional[str]:
    """Get currently authenticated username"""
    return st.session_state.get('username')


def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)


def init_session_state():
    """Initialize session state for authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None


def create_password_hash(password: str) -> str:
    """
    Utility function to create password hash for secrets.toml
    
    Usage:
        hash = create_password_hash("mypassword")
        Add to secrets.toml:
        [passwords]
        username = "hash_value"
    """
    return hash_password(password)


# Example secrets.toml format:
"""
OPENROUTER_API_KEY = "your_api_key_here"

[passwords]
admin = "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"  
user1 = "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8"  
demo = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"  
"""
