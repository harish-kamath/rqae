'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

const DATASETS = ['monology_pile']; // Add more datasets as needed
const EXAMPLES_PER_PAGE = 10;

interface Example {
  text: string;
  id: number;
}

export default function Home() {
  const router = useRouter();
  const [selectedDataset, setSelectedDataset] = useState(DATASETS[0]);
  const [examples, setExamples] = useState<Example[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchMoreExamples = async () => {
    setLoading(true);
    try {
      const newExamples: Example[] = [];
      for (let i = 0; i < EXAMPLES_PER_PAGE; i++) {
        const response = await fetch(
          `https://harish-kamath--rqae-server-fastapi-app.modal.run/stream_text?dataset_name=${selectedDataset}`
        );
        const data = await response.json();
        newExamples.push({
          text: data.text,
          id: data.id
        });
      }
      setExamples(prev => [...prev, ...newExamples]);
    } catch (error) {
      console.error('Error fetching examples:', error);
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    setExamples([]);
    fetchMoreExamples();
  }, [selectedDataset]);

  return (
    <main className="min-h-screen p-8 bg-gray-50">
      {/* Dataset Selector */}
      <div className="mb-8">
        <label htmlFor="dataset" className="block text-sm font-medium text-gray-700 mb-2">
          Select Dataset
        </label>
        <select
          id="dataset"
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
          className="block w-full max-w-xs px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
        >
          {DATASETS.map(dataset => (
            <option key={dataset} value={dataset}>
              {dataset}
            </option>
          ))}
        </select>
      </div>

      {/* Examples List */}
      <div className="space-y-4">
        {examples.map((example) => (
          <div
            key={example.id}
            className="p-4 bg-white rounded-lg shadow hover:shadow-md transition-shadow"
          >
            <div className="flex justify-between items-start gap-4">
              <p className="text-gray-800 whitespace-pre-wrap font-mono text-sm flex-grow">
                {example.text}
              </p>
              <button
                onClick={() => router.push(`/${example.id}`)}
                className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors flex-shrink-0"
              >
                Analyze
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Show More Button */}
      <div className="mt-8 flex justify-center">
        <button
          onClick={fetchMoreExamples}
          disabled={loading}
          className={`px-4 py-2 rounded-md text-white ${loading
            ? 'bg-indigo-400 cursor-not-allowed'
            : 'bg-indigo-600 hover:bg-indigo-700'
            }`}
        >
          {loading ? 'Loading...' : 'Show More'}
        </button>
      </div>
    </main>
  );
}
