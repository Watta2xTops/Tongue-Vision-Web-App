import Image from "next/image";
import logo from "/public/assets/Denty.png";
import Link from 'next/link';

const HomeCard4 = ({ className = "" }) => {
    return (
        <div
           className={`bg-gradient-to-br from-[#4fa1f2] via-[#74b0f0] to-[#d3eaff] 
            backdrop-blur-md bg-opacity-30 rounded-3xl p-6 shadow-md hover:shadow-blue-300 
            transition-shadow duration-500 ${className}`}
        >
            <h2 className="text-2xl font-semibold text-white mb-2">Have any questions? Ask Denty!</h2>
            <div className="flex justify-center -mt-8">
                <Image
                    src={logo}
                    alt="Denty the Assistant"
                    width={220}
                    style={{ height: "auto" }}
                />
            </div>
            <div className="flex justify-center">
                <Link href="/chatbot">
                    <button className="mt-1 w-72 px-4 py-2 bg-white/20 text-white rounded-3xl hover:bg-[#608cc4]/40 transition-colors duration-200">
                        Chat with Denty!
                    </button>
                </Link>
            </div>
        </div>
    );
  };
  
  export default HomeCard4;
  