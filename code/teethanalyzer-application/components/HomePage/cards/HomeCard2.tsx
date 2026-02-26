import Link from 'next/link';

const HomeCard2 = ({ className = "" }) => {
  return (
    <div
      className={`bg-[url('/assets/Dental%20Desk%20Image%20V8.png')] bg-cover bg-center
        backdrop-blur-md bg-opacity-30 rounded-3xl p-6 shadow-md hover:shadow-blue-300
        transition-shadow duration-500 text-white ${className}`}
    >
      <h2 className="text-2xl font-semibold mb-2">Check your Teeth's Health</h2>
      <Link href="/scan">
        <button className="mt-2 px-4 py-2 bg-white/20 text-white rounded-3xl hover:bg-[#608cc4]/40 transition-colors duration-200">
          Activate Teeth Scan
        </button>
      </Link>
    </div>
  );
};

export default HomeCard2;
