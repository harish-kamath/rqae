"use client"

import TokenViewer from '@/components/tokenviewer';
import { FC, useState, useEffect } from 'react';

interface FeaturePageProps {
    params: {
        modelid: string;
        id: string;
    };
}

interface SequenceData {
    sequence: string[];
    activations: number[];
}

const FeaturePage: FC<FeaturePageProps> = ({ params }) => {
    const { modelid, id } = params;
    const [data, setData] = useState<SequenceData[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

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

                const reader = response.body?.getReader();
                if (!reader) {
                    throw new Error('Failed to get reader from response');
                }

                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');

                    try {
                        const jsonData = JSON.parse(lines[0]);
                        setData(prevData => [...prevData, jsonData]);
                        setLoading(false);
                    } catch (e) {
                        console.error('Error parsing JSON:', e);
                    }

                    buffer = lines[lines.length - 1];
                }

            } catch (error) {
                console.error('Error fetching data:', error);
                setError('Failed to fetch data. Please try again later.');
                setLoading(false);
            }
        };

        fetchData();
    }, [modelid, id]);

    if (loading) {
        return <div>Loading...</div>;
    }

    if (error) {
        return <div>Error: {error}</div>;
    }

    return (
        <div>
            <h1>Model ID: {modelid}</h1>
            <h2>Feature ID: {id}</h2>
            {data.map((item, index) => (
                <TokenViewer key={index} texts={item.sequence} activations={item.activations} shorthand={true} />
            ))}
        </div>
    );
};

export default FeaturePage;
