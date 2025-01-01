'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

const DATASETS = ['monology_pile']; // Add more datasets as needed
const EXAMPLES_PER_PAGE = 10;

interface Example {
  text: string;
  id: number;
}

const HighlightedText = ({ text, query }: { text: string[]; query: string }) => {
  if (!query.trim()) return <>{text.join('')}</>;

  const parts = text.join('').split(new RegExp(`(${query})`, 'gi'));
  return (
    <>
      {parts.map((part, i) =>
        part.toLowerCase() === query.toLowerCase() ? (
          <span key={i} className="bg-yellow-200">{part}</span>
        ) : (
          part
        )
      )}
    </>
  );
};

export default function Home() {
  const router = useRouter();
  const [selectedDataset, setSelectedDataset] = useState(DATASETS[0]);
  const [examples, setExamples] = useState<Example[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);

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

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      const response = await fetch(
        `https://harish-kamath--rqae-server-fastapi-app.modal.run/search_text?query=${encodeURIComponent(searchQuery)}&dataset_name=${selectedDataset}&limit=100`
      );
      const data = await response.json();
      if (data.success) {
        setExamples(data.results);
      }
    } catch (error) {
      console.error('Error searching:', error);
    } finally {
      setIsSearching(false);
    }
  };

  // Initial load
  useEffect(() => {
    if (!searchQuery) {
      setExamples([]);
      fetchMoreExamples();
    }
  }, [selectedDataset]);

  return (
    <main className="min-h-screen p-8 bg-gray-50">
      {/* Dataset Selector and Search */}
      <div className="mb-8 space-y-4">
        <div>
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

        {/* Search Box */}
        <div className="flex gap-2 max-w-2xl">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            placeholder="Search for specific text..."
            className="flex-grow px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          />
          <button
            onClick={handleSearch}
            disabled={isSearching || !searchQuery.trim()}
            className={`px-4 py-2 rounded-md text-white ${isSearching || !searchQuery.trim()
              ? 'bg-indigo-400 cursor-not-allowed'
              : 'bg-indigo-600 hover:bg-indigo-700'
              }`}
          >
            {isSearching ? 'Searching...' : 'Search'}
          </button>
        </div>
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
                <HighlightedText text={example.text} query={searchQuery} />
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

      {/* Show More Button - Only show when not searching */}
      {!searchQuery && (
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
      )}
    </main>
  );
}
