'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';

interface CacheCheckResult {
    exists: boolean;
    layers: number[];
}

interface TokenSamples {
    top: {
        indices: number[];
        intensities: number[];
        texts: string[][];
    };
    middle: {
        indices: number[];
        intensities: number[];
        texts: string[][];
    };
    bottom: {
        indices: number[];
        intensities: number[];
        texts: string[][];
    };
}

interface TokenDetails {
    cacheExists: boolean;
    layers: number[];
    selectedLayer?: number;
    samples?: TokenSamples;
    isGenerating?: boolean;
    generationProgress?: number;
}

// Helper function to normalize intensities for a sequence
const normalizeIntensities = (intensities: number[]) => {
    const min = Math.min(...intensities);
    const max = Math.max(...intensities);
    const range = max - min;
    return intensities.map(i => range > 0 ? (i - min) / range : 0);
};

// Helper function to convert intensity to color
const getIntensityColor = (intensity: number, shouldShow: boolean) => {
    if (!shouldShow) return 'transparent';
    // Use a gold color (rgb(234, 179, 8)) with reduced max opacity
    return `rgba(234, 179, 8, ${intensity * 0.9})`;
};

// Helper function to determine if intensity should be shown
const shouldShowIntensity = (normalizedIntensity: number) => {
    // Only show color for top 20% of intensities
    return normalizedIntensity > 0.8;
};

const SampleCategory = ({
    title,
    samples,
    intensities,
    texts
}: {
    title: string;
    samples: number[];
    intensities: number[][];
    texts: string[][]
}) => {
    return (
        <div className="space-y-2">
            <h3 className="font-semibold text-gray-700">{title}</h3>
            <div className="space-y-3">
                {samples.map((sample, sampleIdx) => {
                    // Get the sequence's intensities and normalize them
                    const sequenceIntensities = intensities[sampleIdx];
                    const normalizedIntensities = normalizeIntensities(sequenceIntensities);
                    const sampleTokens = texts[sampleIdx][1]; // Get the tokens for this sample

                    return (
                        <div key={sample} className="p-3 bg-gray-50 rounded">
                            <div className="text-xs text-gray-500 mb-1">Sample {sampleIdx + 1} (ID: {sample})</div>
                            <div className="font-mono text-sm flex flex-wrap">
                                {sampleTokens.map((token, tokenIdx) => {
                                    const normalizedIntensity = normalizedIntensities[tokenIdx];
                                    return (
                                        <span
                                            key={tokenIdx}
                                            style={{
                                                backgroundColor: getIntensityColor(
                                                    normalizedIntensity,
                                                    shouldShowIntensity(normalizedIntensity)
                                                ),
                                                padding: '0 2px',
                                                margin: '0 1px',
                                                borderRadius: '2px',
                                                display: 'inline-block',
                                                transition: 'background-color 0.15s ease',
                                            }}
                                            title={`Intensity: ${sequenceIntensities[tokenIdx].toFixed(3)} (Normalized: ${normalizedIntensity.toFixed(3)})`}
                                        >
                                            {token}
                                        </span>
                                    );
                                })}
                            </div>
                            <div className="mt-2 text-xs text-gray-500">
                                Max Intensity: {Math.max(...sequenceIntensities).toFixed(3)},
                                Min Intensity: {Math.min(...sequenceIntensities).toFixed(3)}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default function ExampleDetails() {
    const params = useParams();
    const [text, setText] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const [loadingCache, setLoadingCache] = useState(true);
    const [selectedToken, setSelectedToken] = useState<number | null>(null);
    const [tokenDetails, setTokenDetails] = useState<TokenDetails | null>(null);
    const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
    const [loadingSamples, setLoadingSamples] = useState(false);
    const [generatingCache, setGeneratingCache] = useState(false);

    // Fetch text immediately
    useEffect(() => {
        const fetchText = async () => {
            try {
                const response = await fetch(
                    `https://harish-kamath--rqae-server-fastapi-app.modal.run/get_text_by_id?idx=${params.id}&dataset_name=monology_pile`
                );
                const textData = await response.json();
                setText(textData.text);
                setLoading(false);
            } catch (error) {
                console.error('Error fetching text:', error);
                setLoading(false);
            }
        };

        fetchText();
    }, [params.id]);

    // Check cache availability
    useEffect(() => {
        const fetchCacheData = async () => {
            try {
                const cacheResponse = await fetch(
                    `https://harish-kamath--rqae-server-fastapi-app.modal.run/check_cache?idx=${params.id}&dataset_name=monology_pile`
                );
                const cacheData = await cacheResponse.json();

                setTokenDetails({
                    cacheExists: cacheData.exists,
                    layers: cacheData.layers,
                });
            } catch (error) {
                console.error('Error fetching cache data:', error);
            } finally {
                setLoadingCache(false);
            }
        };

        if (!loading && text.length > 0) {
            fetchCacheData();
        }
    }, [params.id, text, loading]);

    // Fetch samples when token and layer are selected
    useEffect(() => {
        const fetchSamples = async () => {
            if (selectedToken === null || selectedLayer === null) return;

            setLoadingSamples(true);
            try {
                const response = await fetch(
                    `https://harish-kamath--rqae-server-fastapi-app.modal.run/get_token_samples?idx=${params.id}&token_position=${selectedToken}&layer=${selectedLayer}&dataset_name=monology_pile`
                );
                const samples = await response.json();
                setTokenDetails(prev => prev ? {
                    ...prev,
                    selectedLayer,
                    samples
                } : null);
            } catch (error) {
                console.error('Error fetching samples:', error);
            } finally {
                setLoadingSamples(false);
            }
        };

        fetchSamples();
    }, [params.id, selectedToken, selectedLayer]);

    // Function to handle cache generation
    const generateCache = async () => {
        if (!tokenDetails || generatingCache) return;

        setGeneratingCache(true);
        setTokenDetails(prev => prev ? { ...prev, isGenerating: true, generationProgress: 0 } : null);

        try {
            const response = await fetch(
                `https://harish-kamath--rqae-server-fastapi-app.modal.run/get_samples?idx=${params.id}&dataset_name=monology_pile`
            );

            const reader = response.body?.getReader();
            if (!reader) throw new Error('No reader available');

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                // Parse the streamed data
                const text = new TextDecoder().decode(value);
                const lines = text.trim().split('\n');

                // Update progress based on the number of layers processed
                const lastLine = JSON.parse(lines[lines.length - 1]);
                const progress = lastLine.layer / 128; // Assuming max layer is 128

                setTokenDetails(prev => prev ? {
                    ...prev,
                    generationProgress: progress
                } : null);
            }

            // Refresh cache status after generation
            const cacheResponse = await fetch(
                `https://harish-kamath--rqae-server-fastapi-app.modal.run/check_cache?idx=${params.id}&dataset_name=monology_pile`
            );
            const cacheData = await cacheResponse.json();

            setTokenDetails({
                cacheExists: cacheData.exists,
                layers: cacheData.layers,
                isGenerating: false
            });
        } catch (error) {
            console.error('Error generating cache:', error);
            setTokenDetails(prev => prev ? {
                ...prev,
                isGenerating: false,
                generationProgress: 0
            } : null);
        } finally {
            setGeneratingCache(false);
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen p-8 flex items-center justify-center">
                <div className="text-gray-600">Loading...</div>
            </div>
        );
    }

    return (
        <main className="min-h-screen p-8 bg-gray-50">
            {/* Text Display */}
            <div className="mb-8 p-6 bg-white rounded-lg shadow">
                <h1 className="text-xl font-semibold text-gray-800 mb-4">Example {params.id}</h1>
                <div className="text-gray-800 font-mono text-sm">
                    {text.map((token, idx) => (
                        <span
                            key={idx}
                            onClick={() => setSelectedToken(idx)}
                            className={`cursor-pointer px-0.5 rounded ${selectedToken === idx
                                ? 'bg-indigo-100 border-b-2 border-indigo-500'
                                : 'hover:bg-gray-100'
                                }`}
                        >
                            {token}
                        </span>
                    ))}
                </div>
            </div>

            {/* Token Details Section */}
            {selectedToken !== null && (
                <div className="p-6 bg-white rounded-lg shadow">
                    <h2 className="text-lg font-semibold text-gray-800 mb-4">
                        Token Details - Position {selectedToken}
                    </h2>
                    <div className="space-y-4">
                        <div className="text-gray-600">
                            Selected token: <span className="font-mono">{text[selectedToken]}</span>
                        </div>
                        {loadingCache ? (
                            <div className="text-gray-600">Loading cache data...</div>
                        ) : tokenDetails && (
                            <div className="space-y-4">
                                <div className="text-gray-600">
                                    Cache Status: {tokenDetails.cacheExists ? 'Available' : 'Not Available'}
                                </div>
                                {tokenDetails.cacheExists ? (
                                    <>
                                        <div className="flex items-center gap-4">
                                            <label className="text-gray-600">Select Layer:</label>
                                            <select
                                                value={selectedLayer || ''}
                                                onChange={(e) => setSelectedLayer(e.target.value ? parseInt(e.target.value) : null)}
                                                className="px-3 py-1 border border-gray-300 rounded-md"
                                            >
                                                <option value="">Choose a layer</option>
                                                {tokenDetails.layers.map(layer => (
                                                    <option key={layer} value={layer}>
                                                        Layer {layer}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                        {loadingSamples ? (
                                            <div className="text-gray-600">Loading samples...</div>
                                        ) : tokenDetails.samples && (
                                            <div className="space-y-6">
                                                <SampleCategory
                                                    title="Top Examples"
                                                    samples={tokenDetails.samples.top.indices}
                                                    intensities={tokenDetails.samples.top.intensities}
                                                    texts={tokenDetails.samples.top.texts}
                                                />
                                                <SampleCategory
                                                    title="Middle Examples"
                                                    samples={tokenDetails.samples.middle.indices}
                                                    intensities={tokenDetails.samples.middle.intensities}
                                                    texts={tokenDetails.samples.middle.texts}
                                                />
                                                <SampleCategory
                                                    title="Bottom Examples"
                                                    samples={tokenDetails.samples.bottom.indices}
                                                    intensities={tokenDetails.samples.bottom.intensities}
                                                    texts={tokenDetails.samples.bottom.texts}
                                                />
                                            </div>
                                        )}
                                    </>
                                ) : (
                                    <div className="space-y-4">
                                        {tokenDetails.isGenerating ? (
                                            <div className="space-y-2">
                                                <div className="text-gray-600">
                                                    Generating cache... {Math.round(tokenDetails.generationProgress! * 100)}%
                                                </div>
                                                <div className="w-full bg-gray-200 rounded-full h-2.5">
                                                    <div
                                                        className="bg-indigo-600 h-2.5 rounded-full transition-all duration-300"
                                                        style={{ width: `${tokenDetails.generationProgress! * 100}%` }}
                                                    ></div>
                                                </div>
                                            </div>
                                        ) : (
                                            <button
                                                onClick={generateCache}
                                                disabled={generatingCache}
                                                className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 transition-colors disabled:bg-indigo-400"
                                            >
                                                Generate Cache
                                            </button>
                                        )}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            )}
        </main>
    );
} 