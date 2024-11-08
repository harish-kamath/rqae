"use client"

import TokenViewer from '@/components/tokenviewer';
import { FC, useState, useEffect } from 'react';

interface FeaturePageProps {
    params: {
        modelid: string;
        id: string;
    };
}

interface SubsetData {
    sequences: string[][];
    activations: number[][];
    explanation: string;
    scores: Record<string, number>;
}

interface FeatureData {
    [subset: string]: SubsetData;
}

const FeaturePage: FC<FeaturePageProps> = ({ params }) => {
    const { modelid, id } = params;
    const [features, setFeatures] = useState<Record<number, FeatureData>>({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [activationPercentile, setActivationPercentile] = useState(99);
    const [selectedFeature, setSelectedFeature] = useState<number>(0);
    const [showSettings, setShowSettings] = useState(false);
    const [expandedSubsets, setExpandedSubsets] = useState<Record<string, boolean>>({});

    useEffect(() => {
        const fetchData = async () => {
            if (!modelid || !id) {
                return;
            }
            try {
                const response = await fetch(`https://harish-kamath--rqae-server-feature-web-dev.modal.run?model_id=${modelid}&id=${id}`);
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const jsonString = await response.json();
                const data = JSON.parse(jsonString);
                setFeatures(data);
                setSelectedFeature(Object.keys(data)[0] as unknown as number);
                setLoading(false);
            } catch (error) {
                console.error('Error fetching data:', error);
                setError('Failed to fetch data. Please try again later.');
                setLoading(false);
            }
        };

        fetchData();
    }, [modelid, id]);

    useEffect(() => {
        // Set to highest key in features
        setSelectedFeature(Math.max(...Object.keys(features).map(Number)));
    }, [features]);

    if (loading) {
        return <div>Loading...</div>;
    }

    if (error) {
        return <div>Error: {error}</div>;
    }

    const currentFeature = features[selectedFeature];
    const firstSubset = Object.values(currentFeature)[0];

    const toggleSubset = (subsetName: string) => {
        setExpandedSubsets(prev => ({
            ...prev,
            [subsetName]: !prev[subsetName]
        }));
    };

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Navigation Bar */}
            <nav className="bg-white shadow-sm fixed w-full top-0 z-10">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex flex-col justify-center">
                    <div className="text-xl font-medium">{modelid}</div>
                    <div className="text-sm text-gray-500">{id}</div>
                    <div className="absolute right-4 flex gap-2">
                        <button
                            disabled={id === '000000'}
                            className={`px-3 py-1 rounded text-sm ${id === '000000' ? 'bg-gray-200' : 'bg-blue-500 text-white hover:bg-blue-600'}`}
                            onClick={() => {
                                const prevId = String(parseInt(id) - 1).padStart(6, '0');
                                window.location.href = `/feature/${modelid}/${prevId}`;
                            }}
                        >
                            Previous
                        </button>
                        <button
                            className="px-3 py-1 rounded text-sm bg-blue-500 text-white hover:bg-blue-600"
                            onClick={() => {
                                const nextId = String(parseInt(id) + 1).padStart(6, '0');
                                window.location.href = `/feature/${modelid}/${nextId}`;
                            }}
                        >
                            Next
                        </button>
                    </div>
                </div>
            </nav>

            <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20">
                {/* Feature Selector */}
                {Object.keys(features).length > 1 && (
                    <div className="mb-8 border rounded-lg p-4 bg-white">
                        <div className="mb-4">
                            <h3 className="text-sm font-medium text-gray-700 mb-2">Features with Explanations/Metrics</h3>
                            <div className="space-x-4">
                                {Object.keys(features).map((featureKey) => {
                                    const feat = Object.values(features[parseInt(featureKey)])[0];
                                    if (feat.explanation || (feat.scores && Object.keys(feat.scores).length > 0)) {
                                        return (
                                            <button
                                                key={featureKey}
                                                className={`px-3 py-1 rounded transition-colors ${parseInt(featureKey) === selectedFeature
                                                    ? 'text-blue-600 font-medium'
                                                    : 'text-gray-500 hover:bg-gray-100'
                                                    }`}
                                                onClick={() => setSelectedFeature(parseInt(featureKey))}
                                            >
                                                Feature {featureKey}
                                            </button>
                                        );
                                    }
                                    return null;
                                })}
                            </div>
                        </div>
                        <div>
                            <h3 className="text-sm font-medium text-gray-700 mb-2">Other Features</h3>
                            <div className="flex flex-wrap gap-2">
                                {Object.keys(features).map((featureKey) => {
                                    const feat = Object.values(features[parseInt(featureKey)])[0];
                                    if (!feat.explanation && (!feat.scores || Object.keys(feat.scores).length === 0)) {
                                        return (
                                            <button
                                                key={featureKey}
                                                className={`text-sm transition-colors ${parseInt(featureKey) === selectedFeature
                                                    ? 'text-blue-600 font-medium'
                                                    : 'text-gray-400 hover:text-gray-600'
                                                    }`}
                                                onClick={() => setSelectedFeature(parseInt(featureKey))}
                                            >
                                                {featureKey}
                                            </button>
                                        );
                                    }
                                    return null;
                                })}
                            </div>
                        </div>
                    </div>
                )}

                {/* Feature Content */}
                <div className="space-y-6">
                    {/* Explanation and Metrics Row */}
                    <div className="flex gap-4">
                        {firstSubset.scores && Object.keys(firstSubset.scores).length > 0 && (
                            <div className="bg-white rounded-lg shadow p-4 w-1/4">
                                <h3 className="font-medium text-gray-900 mb-4">Metrics</h3>
                                <table className="w-full">
                                    <tbody>
                                        {Object.entries(firstSubset.scores).map(([key, value]) => (
                                            <tr key={key}>
                                                <td className="py-1 text-gray-600">{key}</td>
                                                <td className="py-1 text-right font-medium">{value}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}

                        {firstSubset.explanation && (
                            <div className="bg-white rounded-lg shadow p-4 flex-1">
                                <h3 className="font-medium text-gray-900 mb-2">Explanation</h3>
                                <p className="text-gray-600">{firstSubset.explanation}</p>
                            </div>
                        )}
                    </div>

                    {/* Settings */}
                    <div className="text-sm text-gray-600">
                        <button
                            onClick={() => setShowSettings(!showSettings)}
                            className="flex items-center hover:text-gray-900"
                        >
                            <span className="mr-2">Activation threshold</span>
                            <span className={`transform transition-transform ${showSettings ? 'rotate-180' : ''}`}>▼</span>
                        </button>

                        {showSettings && (
                            <div className="mt-2 space-x-2">
                                <button
                                    className={`px-2 py-1 text-xs rounded ${activationPercentile === 99 ? 'bg-blue-500 text-white' : 'bg-gray-100'}`}
                                    onClick={() => setActivationPercentile(99)}
                                >
                                    Top 1%
                                </button>
                                <button
                                    className={`px-2 py-1 text-xs rounded ${activationPercentile === 90 ? 'bg-blue-500 text-white' : 'bg-gray-100'}`}
                                    onClick={() => setActivationPercentile(90)}
                                >
                                    Top 10%
                                </button>
                                <button
                                    className={`px-2 py-1 text-xs rounded ${activationPercentile === 50 ? 'bg-blue-500 text-white' : 'bg-gray-100'}`}
                                    onClick={() => setActivationPercentile(50)}
                                >
                                    Top 50%
                                </button>
                                <button
                                    className={`px-2 py-1 text-xs rounded ${activationPercentile === 0 ? 'bg-blue-500 text-white' : 'bg-gray-100'}`}
                                    onClick={() => setActivationPercentile(0)}
                                >
                                    All
                                </button>
                            </div>
                        )}
                    </div>

                    {/* Activation Subsets */}
                    <div className="space-y-6">
                        {Object.entries(currentFeature).map(([subsetName, subsetData]) => (
                            <div key={subsetName} className="bg-white rounded-lg shadow p-4">
                                <h3 className="font-medium text-gray-900 mb-4">
                                    {subsetName === "Top Activations" ? "Highest Activation Examples" :
                                        subsetName === "Median Activations" ? "Median Activation Examples" :
                                            "Low/Zero Activation Examples"}
                                </h3>
                                <div className="space-y-2">
                                    {subsetData.sequences.slice(0, 3).map((sequence, index) => (
                                        <TokenViewer
                                            key={index}
                                            texts={sequence}
                                            activations={subsetData.activations[index]}
                                            shorthand={true}
                                            activation_percentile={activationPercentile}
                                        />
                                    ))}
                                    {expandedSubsets[subsetName] && (
                                        <div className="space-y-2">
                                            {subsetData.sequences.slice(3).map((sequence, index) => (
                                                <TokenViewer
                                                    key={index + 3}
                                                    texts={sequence}
                                                    activations={subsetData.activations[index + 3]}
                                                    shorthand={true}
                                                    activation_percentile={activationPercentile}
                                                />
                                            ))}
                                        </div>
                                    )}
                                    {subsetData.sequences.length > 3 && (
                                        <button
                                            onClick={() => toggleSubset(subsetName)}
                                            className="mt-2 text-sm text-blue-500 hover:text-blue-600"
                                        >
                                            {expandedSubsets[subsetName] ? 'Show less' : 'Show more'}
                                        </button>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </main>
        </div>
    );
};

export default FeaturePage;
