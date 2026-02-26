// context/PredictionContext.tsx
import { createContext, useContext, useState, ReactNode } from "react";

type PredictionContextType = {
  predictionResult: string | null;
  setPredictionResult: (result: string | null) => void;
};

const PredictionContext = createContext<PredictionContextType | undefined>(undefined);

export const PredictionProvider = ({ children }: { children: ReactNode }) => {
  const [predictionResult, setPredictionResult] = useState<string | null>(null);

  return (
    <PredictionContext.Provider value={{ predictionResult, setPredictionResult }}>
      {children}
    </PredictionContext.Provider>
  );
};

export const usePrediction = () => {
  const context = useContext(PredictionContext);
  if (!context) {
    throw new Error("usePrediction must be used within a PredictionProvider");
  }
  return context;
};
