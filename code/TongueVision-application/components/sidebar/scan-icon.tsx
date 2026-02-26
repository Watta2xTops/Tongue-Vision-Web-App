import Image from "next/image";
import Link from "next/link";
import logo from "/public/assets/Scan Icon.png";
import { JSX } from "react";

const ScanIcon = ({ isActive = false }: { isActive?: boolean }): JSX.Element => {
  return (
    <Link
      href="/scan"
      className={`group relative w-12 h-12 mx-auto mb-2 rounded-xl cursor-pointer transition-colors overflow-hidden ${
        isActive ? "bg-blue-50" : "hover:bg-blue-50"
      }`}
    >
      <div className="absolute inset-0 z-0 rounded-xl transition-colors" />
      <Image
        src={logo}
        alt="Logo: Scan Icon"
        className="object-contain z-10 relative"
        priority
      />
    </Link>
  );
};

export default ScanIcon;
