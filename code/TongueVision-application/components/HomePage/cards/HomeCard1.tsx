import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import Image from "next/image";
import logo from "/public/assets/Clinic-TongueV2.png";
import Link from "next/link";
import { gql, useQuery } from "@apollo/client";
import { usePrediction } from "context/PredictionContext";

const GET_USER_BY_ID = gql`
  query GetUserById($userId: ID!) {
    getUserById(userId: $userId) {
      name
      teeth_status
      scanRecords {
        date
        result
        notes
      }
    }
  }
`;

type ScanRecord = {
  date: string;
  result: string[] | string;
  notes: string[];
};

const HomeCard1 = ({ className = "", metric = 100 }: { className?: string; metric?: number }) => {
  const { data: session } = useSession();
  const userId = session?.user?.id;

  const { predictionResult } = usePrediction();
  const { data, loading, error, refetch } = useQuery(GET_USER_BY_ID, {
    variables: { userId },
    skip: !userId,
  });

  useEffect(() => {
    if (predictionResult && userId) {
      refetch();
    }
  }, [predictionResult, userId]);

  const name = session?.user?.name || "User";
  const scanRecords: ScanRecord[] = Array.isArray(data?.getUserById?.scanRecords)
    ? data?.getUserById?.scanRecords
    : [];

  let displayResultRaw =
    predictionResult &&
    predictionResult !== "" &&
    predictionResult !== "Invalid image: Please upload a clear image of an actual teeth."
      ? predictionResult
      : scanRecords[scanRecords.length - 1]?.result;

  let displayResult = Array.isArray(displayResultRaw)
    ? displayResultRaw.join(", ")
    : String(displayResultRaw ?? "");

  let recommendedAction = "Go to a dentist";

  if (displayResult?.toLowerCase() === "no diseases detected") {
    displayResult = "None";
    recommendedAction = "Continue current oral hygiene!";
  }

  let filterClass = "";
  if (metric < 50) {
    filterClass = "filter sepia brightness-[30%] contrast-125 hue-rotate-30";
  } else if (metric < 80) {
    filterClass = "filter sepia brightness-75 hue-rotate-15";
  } else if (metric < 90) {
    filterClass = "filter sepia brightness-90";
  }

  const [showHistory, setShowHistory] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<ScanRecord | null>(null);

  const latestRecord = scanRecords[scanRecords.length - 1];
  const firstRecord = scanRecords[0];

  return (
    <div
      className={`bg-gradient-to-tr from-[#6a8ff7] via-[#7eb8f7] to-[#b2ede8]
        backdrop-blur-md bg-opacity-30 rounded-3xl p-6 shadow-md hover:shadow-blue-300
        transition-shadow duration-500 ${className}`}
    >
      <h2 className="text-2xl font-semibold text-white mb-1">Welcome {name}!</h2>
      <h2 className="text-lg font-semibold text-white mb-4">Tongue Assessment Overview</h2>

      {!showHistory ? (
        <div className="flex items-start justify-between gap-4">

          {/* LEFT — Tongue Assessment History */}
          <div className="w-80 h-55 bg-white/20 backdrop-blur-md rounded-3xl p-4 shadow-inner text-white flex-shrink-0">
            <p className="text-lg font-semibold mb-2">Tongue Assessment History</p>
            {firstRecord ? (
              <>
                <p className="text-sm mb-1">
                  Date:{" "}
                  {new Date(Number(firstRecord.date)).toLocaleDateString("en-US", {
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                </p>
                <p className="text-sm mb-1">
                  Result:{" "}
                  {firstRecord.notes[0]?.toLowerCase().includes("healthy")
                    ? "Healthy teeth"
                    : "1 disease detected"}
                </p>
                <p className="text-sm mb-1">Diseases Present:</p>
                <p className="text-sm mb-3 capitalize">
                  {Array.isArray(firstRecord.result)
                    ? firstRecord.result.join(", ")
                    : firstRecord.result}
                </p>
                <div className="flex justify-center">
                  <button
                    className="px-4 py-2 bg-white/30 text-white rounded-3xl hover:bg-[#608cc4]/40 transition-colors duration-200"
                    onClick={() => setShowHistory(true)}
                  >
                    See History
                  </button>
                </div>
              </>
            ) : (
              <p className="text-sm">No scans found.</p>
            )}
          </div>

          {/* CENTER — Logo + Scan Button */}
          <div className="flex flex-col items-center flex-shrink-0 self-end">
            <div className={filterClass}>
              <Image src={logo} alt="Tongue Logo" width={400} height={400} />
            </div>
            <Link href="/scan">
              <button className="mt-5 px-10 py-4 bg-[#a8edd8] text-[#1a6b52] font-semibold rounded-full shadow-md hover:bg-[#7edfc0] transition-colors duration-200 flex items-center gap-2">
                Proceed to Scan
                <span className="text-lg">→</span>
              </button>
            </Link>
          </div>

          {/* RIGHT — Latest Result */}
          <div className="w-80 h-55 bg-white/20 backdrop-blur-md rounded-3xl p-4 shadow-inner text-white flex-shrink-0">
            <p className="text-lg font-semibold mb-2">Latest Result:</p>
            {latestRecord ? (
              <>
                <p className="text-sm mb-1">
                  Date:{" "}
                  {new Date(Number(latestRecord.date)).toLocaleDateString("en-US", {
                    year: "numeric",
                    month: "long",
                    day: "numeric",
                  })}
                </p>
                <p className="text-sm mb-1">
                  Result:{" "}
                  {latestRecord.notes[0] === "Healthy teeth"
                    ? "Healthy teeth"
                    : "Has 1 disease"}
                </p>
                <p className="text-sm mb-1">Diseases Present:</p>
                <p className="text-sm mb-4 capitalize">{displayResult}</p>
                <p className="text-sm font-medium mb-1">Actions to be taken:</p>
                <p className="text-sm">{recommendedAction}</p>
              </>
            ) : (
              <p className="text-sm">No scans found.</p>
            )}
          </div>

        </div>
      ) : (
        // HISTORY VIEW
        <>
          <h2 className="text-2xl font-semibold text-white mb-4">Scan History</h2>
          <div className="bg-white/20 backdrop-blur-md rounded-3xl p-4 shadow-inner text-white mb-4 max-h-[308px] min-h-[308px] relative">
            <p className="text-md font-medium mb-2">Past Scan Results:</p>
            <div className="max-h-64 overflow-y-auto pr-2">
              <ul className="text-md text-white/80 list-disc list-inside ml-4">
                {scanRecords.map((record, index) => (
                  <li
                    key={index}
                    className="cursor-pointer hover:text-white mb-4"
                    onClick={() => setSelectedRecord(record)}
                  >
                    {new Date(Number(record.date)).toLocaleString("en-US", {
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </li>
                ))}
              </ul>
            </div>

            {selectedRecord && (
              <div
                className="fixed inset-0 z-50 flex items-center justify-center backdrop-blur-sm"
                onClick={() => setSelectedRecord(null)}
              >
                <div
                  className="bg-gradient-to-br from-[#4fa1f2] via-[#74b0f0] to-[#66acf4] backdrop-blur-md rounded-3xl p-6 shadow-lg text-white w-11/12 max-w-md relative"
                  onClick={(e) => e.stopPropagation()}
                >
                  <button
                    className="absolute top-2 right-4 text-white text-xl hover:text-red-300"
                    onClick={() => setSelectedRecord(null)}
                  >
                    &times;
                  </button>
                  <p className="text-base font-semibold mb-2">Teeth Scan History</p>
                  <p className="text-sm mb-1">
                    Date:{" "}
                    {new Date(Number(selectedRecord.date)).toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                    })}
                  </p>
                  <p className="text-sm mb-1">
                    Result:{" "}
                    {selectedRecord?.notes?.[0]
                      ? selectedRecord.notes[0].toLowerCase().includes("healthy")
                        ? "Healthy teeth"
                        : "1 disease detected"
                      : "No Notes"}
                  </p>
                  <p className="text-sm mb-1">Diseases Present:</p>
                  <p className="text-sm mb-1">
                    {Array.isArray(selectedRecord.result)
                      ? selectedRecord.result.join(", ")
                      : selectedRecord.result}
                  </p>
                </div>
              </div>
            )}

            <div className="absolute bottom-4 w-full flex justify-center left-0">
              <button
                className="px-4 py-2 bg-white/30 text-white rounded-3xl hover:bg-[#608cc4]/40 transition-colors duration-200"
                onClick={() => setShowHistory(false)}
              >
                Back
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default HomeCard1;