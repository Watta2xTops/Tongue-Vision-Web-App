import Image from "next/image";

const YouTubePreview = ({ videoId }: { videoId: string }) => {
  const thumbnailUrl = `https://img.youtube.com/vi/${videoId}/hqdefault.jpg`;
  const videoUrl = `https://www.youtube.com/watch?v=${videoId}`;

  return (
    <a href={videoUrl} target="_blank" rel="noopener noreferrer">
      <div className="rounded-2xl relative w-[200px] h-[105px] cursor-pointer group overflow-hidden">
        <Image
          src={thumbnailUrl}
          alt="YouTube Video Thumbnail"
          layout="fill"
          objectFit="cover"
          className="scale-110 group-hover:scale-115 transition-transform duration-300"
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <svg
            className="w-12 h-12 text-[#b0d4f4] opacity-100 group-hover:scale-110 transition-transform"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path d="M10 16.5l6-4.5-6-4.5v9z" />
          </svg>
        </div>
      </div>
    </a>
  );
};

export default YouTubePreview;
