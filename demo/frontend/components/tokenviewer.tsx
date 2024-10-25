"use client"

import React, { useState, useMemo } from 'react';

interface InteractiveTextDisplayProps {
    texts: string[];
    actions?: (() => void)[];
    highlightToken?: number;
    activations?: number[];
    shorthand?: boolean;
}

const TokenViewer: React.FC<InteractiveTextDisplayProps> = ({ texts, actions, highlightToken, activations, shorthand = false }) => {
    const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
    const [isExpanded, setIsExpanded] = useState(false);
    const [shorthandIndexShift, setShorthandIndexShift] = useState(0);

    const normalizedActivations = useMemo(() => {
        if (!activations || activations.length === 0) return null;
        const min = Math.min(...activations);
        const max = Math.max(...activations);
        return max === min
            ? activations.map(() => 1)
            : activations.map(a => 0.4 + 0.6 * ((a - min) / (max - min)));
    }, [activations]);

    const displayTokens = useMemo(() => {
        const NUM_TOKENS = 24;
        if (!shorthand || isExpanded) {
            setShorthandIndexShift(0);
            return texts.map((text, index) => ({ text, activation: normalizedActivations ? normalizedActivations[index] : 1 }));
        }

        if (!activations) {
            setShorthandIndexShift(0);
            return [{ text: '...', activation: 1 }, ...texts.slice(0, NUM_TOKENS).map((text, index) => ({ text, activation: 1 }))];
        }

        const maxIndex = activations.indexOf(Math.max(...activations));
        const start = Math.max(0, maxIndex - NUM_TOKENS / 2);
        const end = Math.min(texts.length, start + NUM_TOKENS);
        setShorthandIndexShift(start);
        return [
            ...(start > 0 ? [{ text: '...', activation: 1 }] : []),
            ...texts.slice(start, end).map((text, index) => ({
                text,
                activation: normalizedActivations ? normalizedActivations[start + index] : 1
            }))
        ];
    }, [texts, activations, normalizedActivations, shorthand, isExpanded]);

    function full_display(text: string, activation: number) {
        if (text.trim() === '' && text.includes('\n')) {
            if ((!shorthand || isExpanded) && activation == 0.4) {
                return text.split('\n').map((_, index, array) =>
                    index < array.length - 1 ? <React.Fragment key={index}><br /></React.Fragment> : null
                );
            } else {
                return <b>{'\\n'.repeat(text.split('\n').length - 1)}</b>;
            }
        }
        if (activation > 0.4) {
            return <b>{text}</b>;
        }
        return text;
    }

    const toggleExpand = () => {
        setIsExpanded(!isExpanded);
    };

    return (
        <div className="p-4">
            <div className="inline">
                {displayTokens.map(({ text, activation }, index) => {
                    return (
                        <span
                            key={index}
                            className={`cursor-pointer transition-all duration-10 relative ${hoveredIndex === index
                                ? 'rounded px-1 shadow-sm border-2 border-yellow-400'
                                : ''
                                } ${highlightToken === index
                                    ? 'bg-yellow-300 text-black font-bold'
                                    : ''
                                }`}
                            style={{
                                opacity: activation,
                                color: activation > 0.4 ? 'maroon' : 'inherit'
                            }}
                            onMouseEnter={() => setHoveredIndex(index)}
                            onMouseLeave={() => setHoveredIndex(null)}
                            onClick={actions && actions[index] ? actions[index] : undefined}
                        >
                            {full_display(text, activation)}
                            {(!shorthand || isExpanded) && hoveredIndex === index && activations && activations[index] && activations[index] > 0.1 && (
                                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 bg-gray-800 text-white px-2 py-1 rounded text-sm mb-1">
                                    {activations[index].toFixed(4)}
                                </div>
                                // TODO: Figure out why this doesn't work when isExpanded is False
                            )}
                        </span>
                    );
                })}
            </div>
            {(shorthand && !isExpanded) && (
                <>
                    <button
                        onClick={toggleExpand}
                        className="text-yellow-500"
                    >
                        {isExpanded ? 'Contract' : '...Expand'}
                    </button>
                    &nbsp;(Max Value: {Math.max(...activations).toFixed(4)})
                </>
            )}
        </div>
    );
};

export default TokenViewer;