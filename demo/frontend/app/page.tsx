"use client";

import { useState, useEffect } from 'react';
import TokenViewer from "@/components/tokenviewer";

export default function Home() {
  const [sequences, setSequences] = useState<{ id: number; tokens: string[] }[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchSequences = async (count: number) => {
    setLoading(true);
    const newSequences: { id: number; tokens: string[] }[] = [];
    console.log("Fetching sequences", count);
    for (let i = 0; i < count; i++) {
      const response = await fetch('https://harish-kamath--rqae-server-sequence-dev.modal.run/');
      const data = await response.json();
      newSequences.push({ id: data[0], tokens: data[1] });
      setSequences(prev => [...prev, { id: data[0], tokens: data[1] }]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchSequences(10);
  }, []);

  const loadMore = () => {
    fetchSequences(10);
  };

  return (
    <div>
      {sequences.map((seq, index) => (
        <div key={index}>
          <TokenViewer
            texts={seq.tokens}
            actions={seq.tokens.map((_, tokenIndex) => () => {
              window.open(`/document/${seq.id}?token=${tokenIndex}`, '_blank');
            })}
          />
        </div>
      ))}
      {loading ? (
        <p>Loading...</p>
      ) : (
        <button onClick={loadMore} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded">
          Load More
        </button>
      )}
    </div>
  );
}
