import dynamic from "next/dynamic";
import Head from "next/head";

const Chatbot = dynamic(() => import("@/components/ChatbotPage/Chatbot"), {
  ssr: false,
});

export default function ChatbotPage() {
  return (
    <>
      <Head>
        <title>Chatbot - Dental Lens</title>
      </Head>
      {/*
        <div className="flex h-screen" style={{ backgroundColor: "#abd7ff" }}>
      */}

      <div className="flex h-screen bg-gradient-to-br from-[#4fa1f2] via-[#74b0f0] to-[#d3eaff]">
        <Chatbot />
      </div>
    </>
  );
}
