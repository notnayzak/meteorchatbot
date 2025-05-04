# =====================================================================
# Meteor Chatbot  – single‑file version (Jupyter‑friendly)
# =====================================================================
import warnings, random, string, sys
from pathlib import Path

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Setup ────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")

# Download NLTK data once (safe to re‑run)
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

DATA_DIR = Path(".")                     # notebook / script folder
ANS_FILE = DATA_DIR / "answer.txt"
Q_FILE   = DATA_DIR / "chatbot.txt"

if not ANS_FILE.exists() or not Q_FILE.exists():
    raise FileNotFoundError(
        "❌  Put 'answer.txt' and 'chatbot.txt' in the same directory!"
    )

# ─── Load corpora ────────────────────────────────────────────────────
raw_ans = ANS_FILE.read_text(encoding="utf8", errors="ignore").lower()
raw_q   = Q_FILE.read_text(encoding="utf8",  errors="ignore").lower()

sent_tokens_ans = nltk.sent_tokenize(raw_ans)   # answers corpus
sent_tokens_q   = nltk.sent_tokenize(raw_q)     # questions corpus

# ─── NLP helpers ─────────────────────────────────────────────────────
lemmer = nltk.stem.WordNetLemmatizer()
remove_punct = dict((ord(p), None) for p in string.punctuation)

def LemTokens(tokens):
    return [lemmer.lemmatize(t) for t in tokens]

def LemNormalize(text):
    return LemTokens(
        nltk.word_tokenize(text.lower().translate(remove_punct))
    )

# ─── Canned replies ──────────────────────────────────────────────────
INTRO_ANS = [
    "My name is Meteor Bot.",
    "You can call me Meteor Bot or B.O.T.",
    "I'm Meteor Bot, happy to chat!"
]
GREETING_IN  = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_OUT = ("hi", "hey", "hello", "hi there", "hello there")

def greeting(sentence: str):
    for word in sentence.split():
        if word.lower() in GREETING_IN:
            return random.choice(GREETING_OUT)

# ─── TF‑IDF matcher ──────────────────────────────────────────────────
def generate_response(user_msg: str, corpus):
    """Return best‑matching sentence from *corpus* or fallback apology."""
    corpus.append(user_msg)
    vec   = TfidfVectorizer(tokenizer=LemNormalize, stop_words=None)
    tfidf = vec.fit_transform(corpus)
    sims  = cosine_similarity(tfidf[-1], tfidf)
    
    idx  = sims.argsort()[0][-2]   # index of best match
    flat = sims.flatten(); flat.sort()
    score = flat[-2]               # similarity of best match
    
    corpus.pop()                   # clean up
    if score == 0:
        return "I'm sorry, I didn't understand that."
    return corpus[idx]

# ─── Public chat() function ──────────────────────────────────────────
def chat(user_msg: str) -> str:
    """Front‑door wrapper combining canned and TF‑IDF replies."""
    u = user_msg.strip().lower()
    if not u:
        return "Please say something 🙂"
    
    # quick exits / canned answers
    if u in ("bye", "goodbye", "see you"):
        return "Bye! Take care."
    if u in ("thanks", "thank you", "thx"):
        return "You're welcome."
    if u in ("how are you", "how r u", "how're you",
             "how's it going", "how's everything"):
        return "I'm fine, thank you for asking!"
    if "your name" in u:
        return random.choice(INTRO_ANS)
    
    g = greeting(u)
    if g:
        return g
    
    # dedicated corpus for “module” questions (kept from original code)
    if "module" in u:
        return generate_response(u, sent_tokens_q.copy())
    
    # default: answer corpus
    return generate_response(u, sent_tokens_ans.copy())

# ─── Simple CLI (works inside a notebook cell) ───────────────────────
def run_cli():
    print("Meteor Bot — type 'bye' to quit")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not user:
            continue
        reply = chat(user)
        print("Bot:", reply)
        if user.lower() in ("bye", "goodbye", "see you"):
            break

# ─── Optional Tkinter GUI ────────────────────────────────────────────
def launch_gui():
    """Start a simple Tkinter chat window (runs only on desktop systems)."""
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except ImportError:
        print("Tkinter not available on this system.")
        return
    
    class ChatGUI(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Meteor Chatbot")
            self.geometry("420x470")
            self.configure(bg="#FFC0CB")
            
            # Chat history widget
            self.chat_area = scrolledtext.ScrolledText(
                self, wrap=tk.WORD, state="disabled", font=("Verdana", 10)
            )
            self.chat_area.pack(expand=True, fill="both", padx=6, pady=6)
            
            # Entry + send button
            frame = tk.Frame(self, bg="#FFC0CB")
            frame.pack(fill="x", padx=6, pady=6)
            
            self.var = tk.StringVar()
            entry = tk.Entry(frame, textvariable=self.var, font=("Verdana", 10))
            entry.pack(side="left", fill="x", expand=True, ipady=4)
            entry.bind("<Return>", self.on_send)
            
            btn = tk.Button(frame, text="Send", bg="#9F2B68",
                            fg="black", font=("Verdana", 10, "bold"),
                            command=self.on_send)
            btn.pack(side="left", padx=(6, 0))
            
            self.post("Meteor", "Hello! I'm Meteor Bot. How can I help?")
        
        # ---- helpers ----
        def post(self, who, txt):
            self.chat_area["state"] = "normal"
            self.chat_area.insert("end", f"{who}: {txt}\n")
            self.chat_area["state"] = "disabled"
            self.chat_area.yview("end")
        
        def on_send(self, event=None):
            msg = self.var.get().strip()
            if not msg:
                return
            self.var.set("")
            self.post("You", msg)
            self.after(100, lambda: self.post("Meteor", chat(msg)))
    
    ChatGUI().mainloop()

# ─── Choose how to run (Jupyter or script) ───────────────────────────
if __name__ == "__main__":
    # If executed as a script, open CLI
    run_cli()

# In a notebook, you can:
#   • call chat("hi") directly in a cell, or
#   • run `run_cli()` for a terminal‑style loop, or
#   • run `launch_gui()` to open the Tkinter window
