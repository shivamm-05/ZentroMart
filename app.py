"""ZentroMart Voice & Text AI Support Assistant
Web interface supporting dual-mode interaction (Microphone Voice + Text Chat),
proactive conversation flow, quick-start action chips, and real-time audio replies.

Run: python app.py -> open http://127.0.0.1:7860
"""
import sys
import tempfile
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import gradio as gr

from src.agent import SupportAgent
from src.kb import load_chunks
from src.retriever import Retriever
from src.voice import Voice

retriever = Retriever(load_chunks())
voice = Voice()
TMP = Path(tempfile.mkdtemp())

GREETING_EN = "Hello! I am ZentroMart's AI customer-support assistant. How can I help you with your order, returns, refunds, or product queries today?"


def new_agent():
    return SupportAgent(retriever, channel="voice")


def start_flow():
    """Initial proactive greeting to start conversation in flow."""
    agent = new_agent()
    out_wav = TMP / "greeting.wav"
    ok = voice.speak(GREETING_EN, out_wav)
    chat = [
        {"role": "assistant", "content": GREETING_EN}
    ]
    info = "**Status**: Conversation initialized. State your query, choose a starter, or share your 8-digit order number."
    return agent, chat, (str(out_wav) if ok else None), 0, info, ""


def process_text_turn(user_text, agent, chat, n):
    """Handle text input with spoken audio reply."""
    if not user_text or not str(user_text).strip():
        return agent, chat, None, n, "", ""
    
    text = str(user_text).strip()
    n += 1
    out_wav = TMP / f"reply_{n}.wav"
    
    res = agent.respond(text)
    reply = res.reply
    ok = voice.speak(reply, out_wav)
    
    new_chat = (chat or []) + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": reply}
    ]
    
    src_str = ", ".join(s["id"] for s in res.sources) or "None (General Policy/Greeting)"
    tool_str = ", ".join(c["name"] for c in res.tool_calls) or "None"
    
    dec_color = {
        "answer": "🟢", "clarify": "🔵", "escalate": "🟠", "refuse": "🔴"
    }.get(res.decision, "⚪")
    
    info = (
        f"{dec_color} **Decision**: `{res.decision}`  |  "
        f"📚 **Sources Used**: `{src_str}`  |  "
        f"🛠️ **Tools Executed**: `{tool_str}`"
    )
    return agent, new_chat, (str(out_wav) if ok else None), n, info, ""


def process_voice_turn(audio_path, lang, agent, chat, n):
    """Handle microphone speech input with STT and TTS reply."""
    if audio_path is None:
        return agent, chat, None, n, ""
    
    voice.stt_language = lang
    n += 1
    out_wav = TMP / f"reply_{n}.wav"
    
    t = voice.turn(agent, audio_path, out_wav)
    res = t["result"]
    heard = t["heard"] or "(unclear audio)"
    reply = t["reply"]
    
    new_chat = (chat or []) + [
        {"role": "user", "content": f"🎙️ {heard}"},
        {"role": "assistant", "content": reply}
    ]
    
    info = ""
    if res:
        src_str = ", ".join(s["id"] for s in res.sources) or "None (General Policy/Greeting)"
        tool_str = ", ".join(c["name"] for c in res.tool_calls) or "None"
        dec_color = {
            "answer": "🟢", "clarify": "🔵", "escalate": "🟠", "refuse": "🔴"
        }.get(res.decision, "⚪")
        info = (
            f"{dec_color} **Decision**: `{res.decision}`  |  "
            f"📚 **Sources Used**: `{src_str}`  |  "
            f"🛠️ **Tools Executed**: `{tool_str}`"
        )
    return agent, new_chat, (str(out_wav) if t["audio_ok"] else None), n, info


# ---------------- Build Gradio UI ----------------
with gr.Blocks(title="ZentroMart Voice & Text AI Support", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🛒 ZentroMart Voice & Text AI Support Agent
        **Autonomous, Grounded Customer Support with Gnani Speech AI & Policy Enforcement**
        *Dataset Reference Date: **20 October 2026** | Mon–Sat Business Days (Excl. Holidays)*
        """
    )
    
    # State tracking
    init_agent, init_chat, init_audio, init_count, init_info, _ = start_flow()
    agent_state = gr.State(init_agent)
    count = gr.State(init_count)

    with gr.Row():
        with gr.Column(scale=7):
            chat = gr.Chatbot(value=init_chat, label="Support Conversation", height=380)
            
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder="Type your message, query, or order ID here and press Enter...",
                    label="⌨️ Text Message",
                    scale=9,
                    lines=1
                )
                send_btn = gr.Button("Send 💬", variant="primary", scale=2)
            
            gr.Markdown("#### 💡 Quick Starters (Click to test conversation flow)")
            with gr.Row():
                q1 = gr.Button("📦 Order Details (8978 2837)", size="sm")
                q_verify = gr.Button("🔑 Verify Phone (4533)", size="sm")
                q_detail = gr.Button("📋 What Did I Order?", size="sm")
                q4 = gr.Button("📱 Product Details (ELE-0012)", size="sm")
            with gr.Row():
                q2 = gr.Button("🔄 Return Window", size="sm")
                q3 = gr.Button("💰 Wallet Refund Speed", size="sm")
                q5 = gr.Button("🚫 Cancel Shipped Order", size="sm")
                q6 = gr.Button("👤 Speak to Human Agent", size="sm")

        with gr.Column(scale=5):
            gr.Markdown("### 🎙️ Spoken Voice Interaction")
            lang = gr.Dropdown(
                ["en-IN", "hi-IN", "auto"],
                value="en-IN",
                label="Spoken Language (en-IN handles English & Hinglish, hi-IN handles Hindi)"
            )
            mic = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Click record, speak into mic, then stop to send"
            )
            speech = gr.Audio(
                value=init_audio,
                label="🔊 Agent Voice Reply (Gnani TTS)",
                autoplay=True,
                type="filepath"
            )
            new_call_btn = gr.Button("🔄 Start New Call / Reset", variant="secondary")

    info = gr.Markdown(value=init_info)

    with gr.Accordion("📋 Conversation Flow & Identity Verification Guide", open=False):
        gr.Markdown(
            """
            ### How the Conversation Flow Works:
            1. **Opening (STY-1)**: Agent introduces itself as ZentroMart's AI assistant.
            2. **Order Identification (IDV-1)**: Customer provides an 8-digit order ID (e.g. `8978 2837`).
            3. **Identity Verification Gate (IDV-2 / IDV-3)**:
               - The agent requests **either** the last 4 digits of the registered phone number **or** the 6-digit delivery pincode.
               - Zero order details are exposed prior to successful verification. Max 1 retry allowed.
            4. **Resolution & Policy Grounding**:
               - Verified order facts are retrieved from synthetic CSV tables (`orders.csv`, `products.csv`, `pincodes.csv`).
               - Business-day date math is computed in code using `holidays.csv`.
            5. **Escalation**: High-value damage (> ₹10,000), repeat contacts (≥ 2 in 7 days), or fraud trigger offline `create_ticket`.
            """
        )

    # Wire Text Events
    text_input.submit(
        process_text_turn,
        inputs=[text_input, agent_state, chat, count],
        outputs=[agent_state, chat, speech, count, info, text_input]
    )
    send_btn.click(
        process_text_turn,
        inputs=[text_input, agent_state, chat, count],
        outputs=[agent_state, chat, speech, count, info, text_input]
    )

    # Wire Voice Event
    mic.stop_recording(
        process_voice_turn,
        inputs=[mic, lang, agent_state, chat, count],
        outputs=[agent_state, chat, speech, count, info]
    )

    # Wire Reset Event
    new_call_btn.click(
        start_flow,
        inputs=None,
        outputs=[agent_state, chat, speech, count, info, text_input]
    )

    # Wire Quick Starter Buttons
    q1.click(lambda: "Can you tell me about the details of order 8978 2837?", None, text_input).then(
        process_text_turn, [text_input, agent_state, chat, count], [agent_state, chat, speech, count, info, text_input]
    )
    q_verify.click(lambda: "Registered phone last 4 digits are 4533", None, text_input).then(
        process_text_turn, [text_input, agent_state, chat, count], [agent_state, chat, speech, count, info, text_input]
    )
    q_detail.click(lambda: "Can you tell me about detail of my order?", None, text_input).then(
        process_text_turn, [text_input, agent_state, chat, count], [agent_state, chat, speech, count, info, text_input]
    )
    q4.click(lambda: "Tell me details of product ELE-0012", None, text_input).then(
        process_text_turn, [text_input, agent_state, chat, count], [agent_state, chat, speech, count, info, text_input]
    )
    q2.click(lambda: "What is the return window for electronics?", None, text_input).then(
        process_text_turn, [text_input, agent_state, chat, count], [agent_state, chat, speech, count, info, text_input]
    )
    q3.click(lambda: "How fast are refunds to the ZentroMart Wallet?", None, text_input).then(
        process_text_turn, [text_input, agent_state, chat, count], [agent_state, chat, speech, count, info, text_input]
    )
    q5.click(lambda: "My order has shipped already. Can I cancel it?", None, text_input).then(
        process_text_turn, [text_input, agent_state, chat, count], [agent_state, chat, speech, count, info, text_input]
    )
    q6.click(lambda: "I want to speak with a human support person right now.", None, text_input).then(
        process_text_turn, [text_input, agent_state, chat, count], [agent_state, chat, speech, count, info, text_input]
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT", 7860))
    # On Railway / cloud containers, bind to 0.0.0.0; on local PC, bind to 127.0.0.1
    host = "0.0.0.0" if (os.environ.get("PORT") or os.environ.get("RAILWAY_ENVIRONMENT")) else "127.0.0.1"
    try:
        demo.launch(server_name=host, server_port=port)
    except OSError:
         demo.launch(server_name="127.0.0.1", share=True)