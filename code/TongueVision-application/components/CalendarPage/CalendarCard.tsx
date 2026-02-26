"use client";

import { useEffect, useState } from "react";
import { FaPencilAlt } from "react-icons/fa";
import clsx from "clsx";

const weekdays = ["M", "T", "W", "T", "F", "S", "S"];
const calendarGrid = Array.from({ length: 5 }, (_, row) =>
  Array.from({ length: 7 }, (_, col) => row * 7 + col + 1)
);

export default function CalendarCard() {
  const [isFlipped, setIsFlipped] = useState(false);
  const [date, setDate] = useState(new Date());
  const [generalTasks, setGeneralTasks] = useState<string[]>([]);
  const [newTask, setNewTask] = useState("");

  const [selectedDay, setSelectedDay] = useState<number | null>(null);
  const [tasksByDate, setTasksByDate] = useState<Record<number, string[]>>({});
  const [newDateTask, setNewDateTask] = useState("");

  useEffect(() => {
    const interval = setInterval(() => setDate(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  const day = date.getDate();
  const weekday = date.toLocaleDateString("en-US", { weekday: "long" });
  const month = date.toLocaleDateString("en-US", { month: "long" });
  const year = date.getFullYear();

  return (
    <div className="relative w-full max-w-7xl aspect-[16/9] perspective mx-auto my-2">
      {/* Card Container */}
      <div
        className={clsx(
          "absolute w-full h-full transition-transform duration-1000 transform-style-preserve-3d",
          isFlipped ? "rotate-y-180" : ""
        )}
      >
        {/* Front Side */}
        <div className="absolute w-full h-full rounded-xl bg-white text-[#58a4f4] border-t-20 border-l-20 border-b-20 border-[#58a4f4] backface-hidden overflow-hidden">
          <div className="flex flex-col md:flex-row h-full">
            {/* Calendar Table */}
            <div className="flex-grow p-6 flex items-start justify-start">
              <div className="w-full max-w-sm">
                <table className="w-full text-5xl font-medium">
                  <thead className="text-[#58a4f4]">
                    <tr>
                      {weekdays.map((day, idx) => (
                        <th key={idx} className="py-6 border-b-2 border-[#58a4f4]">
                          {day}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="text-[#58a4f4]">
                    {calendarGrid.map((week, i) => (
                      <tr key={i}>
                        {week.map((dayNum) => (
                          <td
                            key={dayNum}
                            className={`px-9 text-center hover:text-[#1f6bb7] transition-colors duration-300 cursor-pointer ${
                              i === 0 ? "pt-10 pb-5" : "py-5"
                            }`}
                            onClick={() => {
                              if (dayNum <= 31) {
                                setSelectedDay(dayNum);
                                setIsFlipped(true);
                              }
                            }}
                          >
                            {dayNum <= 31 ? dayNum : ""}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Sidebar — Stop card flipping when interacting here */}
            <div
              className="w-full md:w-1/4 bg-[#58a4f4] p-6 rounded-b-lg md:rounded-b-none text-right text-[#ECECE7] relative"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex flex-col h-full items-end">
                {/* Date Info */}
                <div className="text-right mb-6">
                  <div className="text-5xl font-bold">{day}</div>
                  <div className="text-sm text-white">{weekday}</div>
                  <div className="text-xs text-white">
                    {month} / {year}
                  </div>
                </div>

                
              </div>
              <FaPencilAlt className="absolute bottom-2 right-2 text-[#ECECE7] hover:text-[#232227] transition-colors duration-300" />
            </div>
          </div>
        </div>

        {/* Back Side */}
        <div className="absolute w-full h-full bg-[#58a4f4] rounded-lg rotate-y-180 backface-hidden">
          <div className="m-4 p-4 bg-white rounded-lg h-[calc(100%-2rem)] flex flex-col text-[#232227]">
            <h2 className="text-xl font-semibold mb-2">
              Tasks for {month} {selectedDay}, {year}
            </h2>

            <ul className="flex-grow overflow-y-auto list-disc pl-5 text-sm space-y-1 mb-4">
              {(tasksByDate[selectedDay ?? 0] || []).map((task, idx) => (
                <li key={idx}>{task}</li>
              ))}
            </ul>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                if (!selectedDay || !newDateTask.trim()) return;
                setTasksByDate((prev) => ({
                  ...prev,
                  [selectedDay]: [...(prev[selectedDay] || []), newDateTask.trim()],
                }));
                setNewDateTask("");
              }}
              className="flex gap-2 mt-auto"
            >
              <input
                type="text"
                value={newDateTask}
                onChange={(e) => setNewDateTask(e.target.value)}
                placeholder="Add task"
                className="flex-grow px-3 py-2 rounded border text-sm"
              />
              <button
                type="submit"
                className="bg-[#1f6bb7] text-white px-4 py-2 rounded hover:bg-[#145399] transition"
              >
                Add
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
