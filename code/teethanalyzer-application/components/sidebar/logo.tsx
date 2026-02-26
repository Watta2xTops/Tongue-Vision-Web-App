import Image from "next/image";
import Link from "next/link";
import logo from "/public/assets/Dental Lens Logo.png";
import { JSX } from "react";

const Logo = (): JSX.Element => {
  return (
    <div className="relative w-16 h-16 mx-auto mb-4">
      <Image
        src={logo}
        alt="Logo: Dental Lens"
        fill
        className="object-contain"
        priority
      />
    </div>
  );
};

export default Logo;
