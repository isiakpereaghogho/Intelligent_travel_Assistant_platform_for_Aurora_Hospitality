import streamlit as st
import streamlit.components.v1 as components
import uuid
from dotenv import load_dotenv
load_dotenv(override=True)
import requests

#API URL

import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_URL = f"{API_BASE_URL}/chat"

def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

#Session

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

#page config
st.set_page_config(
page_title="Aurora Hotel Customer Assistant Chatbot",
layout="wide",
#initial_sidebar_state="expanded"
)

#Side bar

with st.sidebar:
    #st.sidebar.success("Sidebar is working")
    st.markdown("""
    <div style='padding: 16px 0 24px 0;'>
        <div class='logo-text'><span style='font-size:1.4rem; font-weight:Bold'>AURORA HOTEL</span><span style='color:#52acff;font-size:1.4rem; font-weight:Bold'> CHATBOT</span></div>
        <div style='font-family:Share Tech Mono;font-size:0.8rem;
            color:#DADADA;letter-spacing:1px;margin-top:auto;'>
            You can ask questions about the hotel, its services, and policies.
            </div>
    <div class='cyber-divider'></div>
    """, unsafe_allow_html=True)

    # API Status
    api_online = check_api_health()
    status_color = '#00ff88' if api_online else '#ff4757'
    status_text  = 'You are now online.' if api_online else "Offline, you won't recieve messages"
    st.markdown(f"""
    <div style='margin-bottom:20px;'>
        <span style='font-family:Share Tech Mono;font-size:0.65rem;
        color:#8892a4;letter-spacing:2px;'>CHAT STATUS</span><br>
        <span style='font-family:Share Tech Mono;font-size:0.85rem;
        color:{status_color};'>● {status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='cyber-divider'></div>", unsafe_allow_html=True)

    guest_type = st.selectbox(
    "Guest Type",
    ["Business", "Leisure", "Family", "Solo", "Couple", "Group", "VIP"],
    index=0)

    loyalty = st.selectbox(
    "Loyalty Tier",
    ["Regular", "Bronze", "Silver", "Gold", "Platinum"],
    index=2)

    city = st.selectbox(
    "City",
    ["New York", "Los Angeles", "Chicago", "Houston", "Sydney", "Melbourne", "Brisbane", "Perth"],
    index=0)

    st.session_state.guest_info = {
    "guest_type": guest_type,
    "loyalty": loyalty,
    "city": city
}

    st.markdown("---")
    st.markdown("<div class='cyber-divider'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:Share Tech Mono;font-size:0.7rem;
    color:#DADADA;letter-spacing:1px;margin-top:auto;'>
    This chatbot is powered by RAG (Retrieval-Augmented Generation) model. 
    </div>
    """, unsafe_allow_html=True)
    #st.write("This chatbot is powered by a RAG (Retrieval-Augmented Generation) model. It can answer questions based on the hotel's policies and information.")

# Main chat interface (HTML)

# Use f-string for the entire HTML block to interpolate Python variables
chat_html = f"""
<!DOCTYPE html>
<html>
<head>

<link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.7.0/css/all.css">

<style>
body, html {{
    height: 70%;
    margin: 0;
    background:rgba(0, 0, 0, 0.04);
}}

.chat{{
    margin: auto;
}}

.card{{
    height: 600px;
    border-radius: 15px;
    background-color: rgba(0, 0, 0, 0.09);
    border:2px solid #52acff;
    overflow:hidden;
}}

/* Dedicated header container, visually separated from the message body */
.chat-header{{
    background-color: rgba(82, 172, 255, 0.08);
    border-bottom: 1px solid rgba(82, 172, 255, 0.35);
    padding: 18px 16px 14px 16px;
}}

.chat-title{{
    text-align:center;
    color:#ffffff;
    font-weight:bold;
    letter-spacing:4px;
    font-size:1.6rem;
}}

.chat-datetime{{
    text-align:center;
    font-family:'Share Tech Mono', monospace;
    font-size:0.75rem;
    color:#8892a4;
    letter-spacing:2px;
    margin-top:6px;
}}

.msg_card_body{{
    overflow-y: auto;
    height: calc(600px - 150px);
}}

.type_msg{{
    background-color: rgba(0, 0, 0, 0.03);
    border:0;
    color: white;
    }}

.msg_container{{
    padding: 10px;
    border-radius:25px;
    background-color:#52acff;
    }}

.msg_container_send{{
    border-radius:25px;
    background-color:#B8DCFF;
    padding:10px;
}}

.user_img{{
    height:60px;
    width:60px;
}}

.user_img_msg{{
    height:35px;
    width:35px;
}}

/* Wraps a bubble (and, for bot messages, its sender label) so
   a timestamp can be placed directly beneath it. */
.message-stack{{
    display:flex;
    flex-direction:column;
    max-width:75%;
}}

.d-flex.justify-content-end .message-stack{{
    align-items:flex-end;
}}

.d-flex.justify-content-start .message-stack{{
    align-items:flex-start;
}}

.msg_sender_name{{
    font-family:'Share Tech Mono', monospace;
    font-size:0.7rem;
    color:#52acff;
    letter-spacing:1px;
    margin-bottom:4px;
    margin-left:6px;
}}

.msg-time{{
    font-family:'Share Tech Mono', monospace;
    font-size:0.65rem;
    color:#8892a4;
    letter-spacing:1px;
    margin-top:4px;
    padding:0 6px;
}}

/* Typing / thinking indicator shown while awaiting a response */
.typing-indicator{{
    display:flex;
    align-items:center;
    gap:5px;
    padding:14px 18px;
}}

.typing-dot{{
    width:7px;
    height:7px;
    border-radius:50%;
    background-color:rgba(0, 0, 0, 0.55);
    animation: typingBounce 1.2s infinite ease-in-out;
}}

.typing-dot:nth-child(2){{
    animation-delay:0.2s;
}}

.typing-dot:nth-child(3){{
    animation-delay:0.4s;
}}

@keyframes typingBounce{{
    0%, 60%, 100% {{ transform: translateY(0); opacity:0.4; }}
    30% {{ transform: translateY(-4px); opacity:1; }}
}}

</style>

</head>

<body>

<div class="container-fluid h-100">
<div class="row justify-content-center h-100">

<div class="col-md-8 col-xl-6 chat">

<div class="card">

<div class="chat-header">
    <div class="chat-title">Aurora Hotel Customer Support</div>
    <div class="chat-datetime" id="liveDateTime">Loading date &amp; time...</div>
</div>

<div id="messageBody" class="card-body msg_card_body">
    <!-- Initial greeting message -->
    <div class="d-flex justify-content-start mb-3">
        <div class="message-stack">
            <div class="msg_sender_name">AuroraBot</div>
            <div class="msg_container">
                Hi Customer, how can I be of help today?
            </div>
        </div>
    </div>
</div>

<div class="card-footer">

<form id="chatForm" class="input-group">

<input type="text" id="msg" placeholder="Enter your message..." class="form-control type_msg"/>

<div class="input-group-append">

<button class="input-group-text send_btn">
<i class="fas fa-location-arrow"></i>
</button>

</div>
</form>

</div>

</div>
</div>
</div>
</div>

<script>

// Live-updating date and time under the header title
function updateDateTime() {{
    const now = new Date();
    const options = {{
        weekday: 'long',
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    }};
    document.getElementById('liveDateTime').textContent = now.toLocaleString('en-US', options);
}}

updateDateTime();
setInterval(updateDateTime, 1000);

// Tracks the timestamp element currently sitting under the
// most recent message, so it can be removed when a newer
// message arrives.
let lastTimeElement = null;

function getFormattedTime() {{
    const now = new Date();
    return now.toLocaleTimeString('en-US', {{ hour: '2-digit', minute: '2-digit' }});
}}

// Removes any existing "last message" timestamp, then appends
// a fresh one under the given message-stack element.
function markAsLastMessage($messageStack) {{
    if (lastTimeElement) {{
        $(lastTimeElement).remove();
    }}
    let $time = $(`<div class="msg-time">${{getFormattedTime()}}</div>`);
    $messageStack.append($time);
    lastTimeElement = $time[0];
}}

// Builds and appends the "AuroraBot is thinking..." indicator,
// shown while the API request is in flight.
function showTypingIndicator() {{
    let typingHtml = `
    <div class="d-flex justify-content-start mb-3" id="typingIndicator">
        <div class="message-stack">
            <div class="msg_sender_name">AuroraBot</div>
            <div class="msg_container typing-indicator">
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
                <span class="typing-dot"></span>
            </div>
        </div>
    </div>`;

    $("#messageBody").append(typingHtml);
    $("#messageBody").scrollTop($("#messageBody")[0].scrollHeight);
}}

function removeTypingIndicator() {{
    $("#typingIndicator").remove();
}}

//Scroll down the page and mark the greeting as the last message
$(document).ready(function() {{
    $("#messageBody").scrollTop(
        $("#messageBody")[0].scrollHeight
        );
    markAsLastMessage($("#messageBody .message-stack").last());
}});

$("#chatForm").on("submit", function(e){{
    e.preventDefault();

    let text = $("#msg").val();

    if(text.trim() === "") return;

    let userHtml = `
    <div class="d-flex justify-content-end mb-3">
        <div class="message-stack">
            <div class="msg_container_send">
                ${{text}}
            </div>
        </div>
    </div>`;

    let $userBlock = $(userHtml);
    $("#messageBody").append($userBlock);
    markAsLastMessage($userBlock.find(".message-stack"));

    $("#msg").val("");

    $("#messageBody").scrollTop(
        $("#messageBody")[0].scrollHeight
    );

    // Show the "thinking" indicator while the request is in flight
    showTypingIndicator();

    // Python f-string interpolation for API_URL and session state variables
    fetch("{API_URL}", {{
        method:"POST",
        headers:{{
            "Content-Type":"application/json"
        }},
        body:JSON.stringify({{
            question:text,
            session_id:"{st.session_state.session_id}",
            guest_type:"{st.session_state.guest_info['guest_type']}",
            loyalty:"{st.session_state.guest_info['loyalty']}",
            city:"{st.session_state.guest_info['city']}"
        }})
    }})

    .then(res => res.json())

    .then(data => {{

        removeTypingIndicator();

        let botHtml = `
        <div class="d-flex justify-content-start mb-3">
            <div class="message-stack">
                <div class="msg_sender_name">AuroraBot</div>
                <div class="msg_container">
                    ${{data.answer}}
                </div>
            </div>
        </div>`;

        let $botBlock = $(botHtml);
        $("#messageBody").append($botBlock);
        markAsLastMessage($botBlock.find(".message-stack"));

        $("#messageBody").scrollTop(
            $("#messageBody")[0].scrollHeight
        );
    }})

    .catch(err => {{
        removeTypingIndicator();

        let errorHtml = `
        <div class="d-flex justify-content-start mb-3">
            <div class="message-stack">
                <div class="msg_sender_name">AuroraBot</div>
                <div class="msg_container">
                    Sorry, something went wrong. Please try again.
                </div>
            </div>
        </div>`;

        let $errorBlock = $(errorHtml);
        $("#messageBody").append($errorBlock);
        markAsLastMessage($errorBlock.find(".message-stack"));

        $("#messageBody").scrollTop(
            $("#messageBody")[0].scrollHeight
        );
    }});

}});

</script>

</body>
</html>
"""

components.html(chat_html, height=700, scrolling=False)