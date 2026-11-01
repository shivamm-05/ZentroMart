import os
import pandas as pd
from openai import OpenAI
from rag_engine import KnowledgeBaseRAG

class ZentroMartAgent:
    def __init__(self, data_dir="Capstone_Project_Dataset_FTTT_2026/data"):
        self.rag = KnowledgeBaseRAG()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Load tables as strings
        self.customers_df = pd.read_csv(f"{data_dir}/customers.csv", dtype=str)
        self.orders_df = pd.read_csv(f"{data_dir}/orders.csv", dtype=str)
        self.order_items_df = pd.read_csv(f"{data_dir}/order_items.csv", dtype=str)
        
        self.system_prompt = (
            "You are ZentroMart's customer support agent. "
            "Reference Date: 20 October 2026. Business days are Mon-Sat (excluding holidays). "
            "Rules:\n"
            "1. First turn: Introduce yourself as an AI assistant.\n"
            "2. Ground all answers solely in the provided context. If unknown, say you don't know.\n"
            "3. Caller Verification: Never disclose order details until customer verifies their phone last 4 digits or pincode.\n"
            "4. Refuse disallowed requests politely (e.g. ₹500 cash compensation).\n"
            "5. If policy requires a human or customer insists, escalate by creating a ticket.\n"
            "6. Speak in short, natural sentences suitable for text-to-speech."
        )
        self.history = [{"role": "system", "content": self.system_prompt}]

    def lookup_order(self, order_id: str, phone_last4: str = None, pincode: str = None):
        order_row = self.orders_df[self.orders_df["order_id"] == order_id]
        if order_row.empty:
            return {"error": "Order not found."}
        
        order = order_row.iloc[0]
        customer = self.customers_df[self.customers_df["customer_id"] == order["customer_id"]].iloc[0]

        # Verify identity
        verified = False
        if phone_last4 and str(customer["phone"]).endswith(phone_last4):
            verified = True
        if pincode and str(order["delivery_pincode"]) == pincode:
            verified = True

        if not verified:
            return {"error": "Authentication required: Please provide your delivery pincode or the last 4 digits of your phone."}

        return {
            "order_id": order["order_id"],
            "status": order["status"],
            "promised_date": order["promised_date"],
            "payment": order["payment"]
        }

    def respond(self, user_text: str):
        docs, metas = self.rag.retrieve(user_text, top_k=2)
        context_str = "\n".join(docs)
        used_sections = [m.get("section_id") for m in metas]

        turn_prompt = f"Knowledge Context:\n{context_str}\n\nCustomer: {user_text}"
        self.history.append({"role": "user", "content": turn_prompt})

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.history,
            temperature=0.0
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply, used_sections