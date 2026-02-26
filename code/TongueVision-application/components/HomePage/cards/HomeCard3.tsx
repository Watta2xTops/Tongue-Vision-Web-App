import Image from "next/image";
import YouTubePreview from "../../ScanPage/videoComponent/preview";
import Link from "next/link";


const HomeCard3 = ({ className = "" }) => {
    return (
        <div className={`bg-gradient-to-br from-[#4fa1f2] via-[#74b0f0] to-[#d3eaff] 
            backdrop-blur-md bg-opacity-30 rounded-3xl p-6 shadow-md hover:shadow-blue-300 
            transition-shadow duration-500 ${className}`}>
            <h2 className="text-2xl font-semibold text-white mb-2">Check out these Health Tips to take better care of your Teeth!</h2>
            
            <div className="flex justify-center item-center gap-5 flex-wrap mt-8 mb-11">
                <YouTubePreview videoId="B35jRf4EKPA" />
                <YouTubePreview videoId="3zL4Hou1P-c" />
                <YouTubePreview videoId="SCP38MrccsI" />
            </div>
            <div className="flex justify-center">
                <Link href="/recommended">
                    <button className="w-48 px-4 py-2 bg-white/20 text-white rounded-3xl hover:bg-[#608cc4]/40 transition-colors duration-200">
                        See More Tips
                    </button>
                </Link>
            </div>
        </div>
    );
  };
  
  export default HomeCard3;
  