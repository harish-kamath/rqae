"use client";

import TokenViewer from '@/components/tokenviewer';
import React, { useState, useEffect } from 'react';
import { useParams, useSearchParams } from 'next/navigation';

const DocumentPage: React.FC = () => {
    const { id } = useParams();
    const searchParams = useSearchParams();
    const [sequenceId, setSequenceId] = useState<number | null>(null);
    const [tokens, setTokens] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const [chosenToken, setChosenToken] = useState<number | undefined>(() => {
        const tokenParam = searchParams.get('token');
        return tokenParam ? parseInt(tokenParam, 10) : undefined;
    });

    useEffect(() => {
        const fetchData = async () => {
            try {
                setLoading(true);
                const response = await fetch(`https://harish-kamath--rqae-server-sequence-dev.modal.run/?idx=${id}`);
                const data = await response.json();
                setSequenceId(data[0]);
                setTokens(data[1]);
            } catch (error) {
                console.error('Error fetching data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [id]);

    return (
        <div>
            {loading ? <p>Loading...</p> : (
                <>
                    <h1>Sequence ID: {sequenceId}</h1>
                    <TokenViewer
                        texts={tokens}
                        actions={tokens.map((_, tokenIndex) => () => setChosenToken(tokenIndex))}
                        highlightToken={chosenToken}
                    />
                </>
            )}
        </div>
    );
};

export default DocumentPage;
