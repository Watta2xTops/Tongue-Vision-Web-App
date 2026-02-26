import ReactMarkdown from "react-markdown";

type ChatMessageProps = {
  message: string;
  sender: "user" | "bot";
};

const ChatMessage = ({ message, sender }: ChatMessageProps) => {
  const isUser = sender === "user";
  return (
    <div className={`flex items-center ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && (
        <div className="mr-2 self-start -mt-1">
          {/* bot avatar */}
          <img src="/assets/Denty.png" alt="Bot" className="w-16 h-auto"/>
        </div>
      )}

      <div
        className={`max-w-lg p-4 rounded-xl text-md whitespace-pre-wrap ${
          isUser
            ? "bg-white text-gray-800 rounded"
            : "bg-white text-gray-800 rounded"
        }`}
      >
        <ReactMarkdown>{message}</ReactMarkdown>
      </div>

      {isUser && (
        <div className="ml-2">
          {/* user avatar */}
          <img src="/assets/User Icon.png" alt="Bot" className="w-16 h-auto"/>
        </div>
      )}
    </div>
  );
};

export default ChatMessage;
