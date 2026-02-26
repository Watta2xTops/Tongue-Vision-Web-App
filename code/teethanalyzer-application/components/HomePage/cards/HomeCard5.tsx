import Image from "next/image";
import logo from "/public/assets/Teeth Goals Image.png";
import Link from 'next/link';

const HomeCard5 = ({ className = "" }) => {
    return (
        <div 
            className={`bg-gradient-to-br from-[#4fa1f2] via-[#74b0f0] to-[#d3eaff] 
            backdrop-blur-md bg-opacity-30 rounded-3xl p-6 shadow-md hover:shadow-blue-300
                transition-shadow duration-500 text-white ${className}`}
        >
            <h2 className="text-2xl font-semibold text-white mb-2">Create your Teeth Goals to achieve Better Oral Health!</h2>
            <div className="flex justify-center -mt-1">
                <Image
                    src={logo}
                    alt="Denty the Assistant"
                    width={119}
                    style={{ height: "auto" }}
                />
            </div>
            <div className="flex justify-center">
                <Link href="/calendar">
                    <button className=" w-72 px-4 py-2 bg-white/20 text-white rounded-3xl hover:bg-[#608cc4]/40 transition-colors duration-200">
                        Create Teeth Goals
                    </button>
                </Link>
            </div>
        </div>   
    );
  };
  
  export default HomeCard5;
  