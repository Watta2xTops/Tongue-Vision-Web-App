import dynamic from "next/dynamic";
import Head from "next/head";

const YouTubeSlider = dynamic(() => import("@/components/RecoPage/YouTubeSlider"), { ssr: false });
//hello
export default function YouTubeSliderPage() {
  return (
    <>
      <Head>
        <title>Recommendations - Dental Lens</title>
      </Head>
      <div style={{ fontFamily: "Segoe UI, Roboto, sans-serif", padding: "1.5rem" }}>
        <YouTubeSlider />
      </div>
    </>
  );
}