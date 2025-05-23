from tkinter import *
import time
from chatBot import chat 

window_size = "400x400"

class ChatInterface(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.master.configure(bg='#FFC0CB')  # window background

        self.tl_bg = "#D8BFD8"
        self.tl_bg2 = "#D8BFD8"
        self.tl_fg = "#000000"
        self.font = "Verdana 10"

        menu = Menu(self.master)
        self.master.config(menu=menu, bd=5)

        # File Menu
        file = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file)
        file.add_command(label="Clear Chat", command=self.clear_chat)
        file.add_command(label="Exit", command=self.chatexit)

        # Text Frame (chat history)
        self.text_frame = Frame(self.master, bd=6)
        self.text_frame.pack(expand=True, fill=BOTH)

        self.text_box_scrollbar = Scrollbar(self.text_frame, bd=0)
        self.text_box_scrollbar.pack(fill=Y, side=RIGHT)

        self.text_box = Text(
            self.text_frame, yscrollcommand=self.text_box_scrollbar.set,
            state=DISABLED, bd=1, padx=6, pady=6, spacing3=8, wrap=WORD,
            bg='white', fg='black', font="Verdana 10", relief=GROOVE,
            width=10, height=1
        )
        self.text_box.pack(expand=True, fill=BOTH)
        self.text_box_scrollbar.config(command=self.text_box.yview)

        # Entry Frame (input area)
        self.entry_frame = Frame(self.master, bd=1)
        self.entry_frame.pack(side=LEFT, fill=BOTH, expand=True)

        self.entry_field = Entry(self.entry_frame, bd=1, justify=LEFT, font="Verdana 10", bg='white')
        self.entry_field.pack(fill=X, padx=6, pady=6, ipady=3)

        # Send Button Frame
        self.send_button_frame = Frame(self.master, bd=0)
        self.send_button_frame.pack(fill=BOTH, padx=6, pady=4)

        self.send_button = Button(
            self.send_button_frame, text="Send", width=7,
            bg="#9F2B68", fg="black", font="Verdana 10 bold", bd=0,
            command=lambda: self.send_message_insert(None),
            activebackground="#F2D2BD", activeforeground="black"
        )
        self.send_button.pack(side=LEFT, ipady=8)
        self.master.bind("<Return>", self.send_message_insert)

        self.last_sent_label(date="No messages sent.")

    def last_sent_label(self, date):
        try:
            self.sent_label.destroy()
        except AttributeError:
            pass
        self.sent_label = Label(
            self.entry_frame, font="Verdana 8 italic", text=date,
            bg=self.tl_bg2, fg="#800080"
        )
        self.sent_label.pack(side=LEFT, fill=X, padx=3)

    def clear_chat(self):
        self.text_box.config(state=NORMAL)
        self.last_sent_label(date="No messages sent.")
        self.text_box.delete(1.0, END)
        self.text_box.config(state=DISABLED)

    def chatexit(self):
        exit()

    def send_message_insert(self, message):
        user_input = self.entry_field.get()
        if not user_input.strip():
            return  # Don't send empty messages

        self.text_box.config(state=NORMAL)
        self.text_box.insert(END, "You : " + user_input + "\n")
        self.text_box.config(state=DISABLED)
        self.text_box.see(END)

        response = chat(user_input)
        self.text_box.config(state=NORMAL)
        self.text_box.insert(END, "Meteor : " + response + "\n")
        self.text_box.config(state=DISABLED)
        self.text_box.see(END)

        self.last_sent_label(time.strftime("Last message sent: %B %d, %Y at %I:%M %p"))
        self.entry_field.delete(0, END)

# Main loop
root = Tk()
chat_ui = ChatInterface(root)
root.geometry(window_size)
root.title("Meteor Chatbot")

# Optional icon (safe fallback)
try:
    root.iconbitmap('i.ico')
except:
    pass

root.mainloop()
