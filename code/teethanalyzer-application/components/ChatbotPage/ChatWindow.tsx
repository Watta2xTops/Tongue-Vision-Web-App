import ChatMessage from "./ChatMessage";
import React, { useEffect, useRef } from "react";

type Message = { sender: "user" | "bot"; text: string };

type ChatWindowProps = {
  messages: Message[];
  isTyping?: boolean;
};

const ChatWindow = ({ messages, isTyping }: ChatWindowProps) => { 
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  return (
    <div className="flex flex-col p-4 space-y-8 overflow-y-auto flex-grow rounded-xl">
      {messages.map((msg, idx) => (
        <ChatMessage key={idx} message={msg.text} sender={msg.sender} />
      ))}

      {isTyping && (
        <div className="flex space-x-1 items-center ml-2">
          <div className="w-2 h-2 bg-white rounded-full animate-bounce" />
          <div className="w-2 h-2 bg-white rounded-full animate-bounce [animation-delay:0.15s]" />
          <div className="w-2 h-2 bg-white rounded-full animate-bounce [animation-delay:0.3s]" />
        </div>
      )}

      <div ref={bottomRef} />
  </div>
  );
};

export default ChatWindow;
