import { useState } from "react";

interface Props {
  onSend: (text: string) => void;
}

const ChatTextBox = ({ onSend }: Props) => {
  const [input, setInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    onSend(input);
    setInput("");
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="flex p-3 bg-white rounded-2xl shadow mt-4"
    >
      <input
        className="flex-grow p-2 outline-none rounded-l-xl"
        placeholder="Ask Denty"
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />
      <button
        type="submit"
        className="bg-[#5cacf4] text-white px-4 py-2 rounded-r-xl hover:bg-[#4396f0]"
      >
        Send
      </button>
    </form>
  );
};

export default ChatTextBox;
