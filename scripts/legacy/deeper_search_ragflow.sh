#!/bin/bash

# RAGFlow Deeper Search Script with DSPy Query Deepening
# Usage: deeper_search_ragflow.sh -k KNOWLEDGE_BASE -n NUMBER_OF_HITS -d DEEPENING_LEVEL "query"
# Example: deeper_search_ragflow.sh -k BASF -n 10 -d 2 "what is a volatility clock"

set -euo pipefail

# Default values
KNOWLEDGE_BASE=""
NUM_HITS=10
DEEPENING_LEVEL=1
QUERY=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.json"

# Function to display usage
usage() {
    echo "Usage: $0 -k KNOWLEDGE_BASE [-n NUM_HITS] [-d DEEPENING_LEVEL] \"query\""
    echo ""
    echo "Options:"
    echo "  -k KNOWLEDGE_BASE     Name of the knowledge base to search (required)"
    echo "  -n NUM_HITS           Number of hits to return (default: 10)"
    echo "  -d DEEPENING_LEVEL    DSPy query refinement level 0-3 (default: 1)"
    echo "                        0 = no deepening, 1 = basic, 2 = gap analysis, 3 = full optimization"
    echo "  -h                    Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -k BASF -n 10 -d 2 \"what is a volatility clock\""
    exit 1
}

# Function to check if required tools are installed
check_dependencies() {
    for cmd in jq curl; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo "Error: $cmd is required but not installed." >&2
            exit 1
        fi
    done
}

# Function to read configuration
read_config() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "Error: Configuration file $CONFIG_FILE not found." >&2
        echo "Please create config.json based on config.json.sample" >&2
        exit 1
    fi
    
    RAGFLOW_BASE_URL=$(jq -r '.RAGFLOW_BASE_URL // empty' "$CONFIG_FILE")
    RAGFLOW_API_KEY=$(jq -r '.RAGFLOW_API_KEY // empty' "$CONFIG_FILE")
    
    if [[ -z "$RAGFLOW_BASE_URL" || -z "$RAGFLOW_API_KEY" ]]; then
        echo "Error: RAGFLOW_BASE_URL and RAGFLOW_API_KEY must be set in $CONFIG_FILE" >&2
        exit 1
    fi
    
    # Remove trailing slash from base URL
    RAGFLOW_BASE_URL=${RAGFLOW_BASE_URL%/}
}

# Function to get dataset ID by name
get_dataset_id() {
    local dataset_name="$1"
    local response
    
    response=$(curl -s -X GET \
        -H "Authorization: Bearer $RAGFLOW_API_KEY" \
        -H "Content-Type: application/json" \
        "$RAGFLOW_BASE_URL/api/v1/datasets" || {
        echo "Error: Failed to fetch datasets" >&2
        exit 1
    })
    
    # Check if response is valid JSON
    if ! echo "$response" | jq . >/dev/null 2>&1; then
        echo "Error: Invalid response from RAGFlow API" >&2
        exit 1
    fi
    
    # Check for API error
    local code
    code=$(echo "$response" | jq -r '.code // 1')
    if [[ "$code" != "0" ]]; then
        local message
        message=$(echo "$response" | jq -r '.message // "Unknown error"')
        echo "Error: API returned error code $code: $message" >&2
        exit 1
    fi
    
    # Find dataset by name (case-insensitive)
    local dataset_id
    dataset_id=$(echo "$response" | jq -r --arg name "${dataset_name,,}" '
        .data[] | select(.name | ascii_downcase | contains($name)) | .id
    ' | head -n1)
    
    if [[ -z "$dataset_id" || "$dataset_id" == "null" ]]; then
        echo "Error: Dataset '$dataset_name' not found." >&2
        echo "Available datasets:" >&2
        echo "$response" | jq -r '.data[]?.name // empty' | sed 's/^/  - /' >&2
        exit 1
    fi
    
    echo "$dataset_id"
}

# Function to search knowledge base with DSPy deepening
search_knowledge_base_deepened() {
    local dataset_id="$1"
    local query="$2"
    local page_size="$3"
    local deepening_level="$4"
    
    local payload
    payload=$(jq -n \
        --arg question "$query" \
        --argjson dataset_ids "[$dataset_id]" \
        --argjson top_k 1024 \
        --argjson similarity_threshold 0.2 \
        --argjson page 1 \
        --argjson page_size "$page_size" \
        --argjson deepening_level "$deepening_level" \
        '{
            question: $question,
            dataset_ids: $dataset_ids,
            top_k: $top_k,
            similarity_threshold: $similarity_threshold,
            page: $page,
            page_size: $page_size,
            deepening_level: $deepening_level
        }')
    
    local response
    response=$(curl -s -X POST \
        -H "Authorization: Bearer $RAGFLOW_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$RAGFLOW_BASE_URL/api/v1/retrieval" || {
        echo "Error: Failed to query RAGFlow API" >&2
        exit 1
    })
    
    # Check if response is valid JSON
    if ! echo "$response" | jq . >/dev/null 2>&1; then
        echo "Error: Invalid response from RAGFlow API" >&2
        exit 1
    fi
    
    # Check for API error
    local code
    code=$(echo "$response" | jq -r '.code // 1')
    if [[ "$code" != "0" ]]; then
        local message
        message=$(echo "$response" | jq -r '.message // "Unknown error"')
        echo "Error: API returned error code $code: $message" >&2
        exit 1
    fi
    
    echo "$response"
}

# Function to format and display results with deepening metadata
format_results_deepened() {
    local response="$1"
    local total_results
    
    total_results=$(echo "$response" | jq -r '.data.total // 0')
    
    if [[ "$total_results" -eq 0 ]]; then
        echo "No results found."
        return
    fi
    
    echo "Found $total_results total results with DSPy query deepening:"
    echo "==========================================================="
    echo ""
    
    # Display deepening metadata if available
    local deepening_info
    deepening_info=$(echo "$response" | jq -r '.metadata.deepening // empty')
    if [[ -n "$deepening_info" && "$deepening_info" != "null" ]]; then
        echo "DSPy Query Deepening Information:"
        echo "---------------------------------"
        echo "$response" | jq -r '
            .metadata.deepening |
            "Original Query: " + .original_query +
            "\nFinal Query: " + .final_query +
            "\nDeepening Level: " + (.deepening_level | tostring) +
            "\nQueries Used: " + (.queries_used | length | tostring) + " iterations"
        '
        echo ""
        
        # Show query evolution
        echo "Query Evolution:"
        echo "$response" | jq -r '.metadata.deepening.queries_used[]' | nl -v0 -s'. '
        echo ""
    fi
    
    # Extract and format chunks with proper HTML tag stripping
    echo "$response" | jq -r '
        .data.chunks[]? | 
        "Result #" + ((.similarity * 100 | floor | tostring) + "% similarity") + ":" +
        "\n" + (.content | gsub("<[^>]*>"; "") | gsub("&nbsp;"; " ") | gsub("&amp;"; "&") | gsub("&lt;"; "<") | gsub("&gt;"; ">") | gsub("&quot;"; "\"") | gsub("&#39;"; "'\''") | gsub("\\s+"; " ") | gsub("^\\s+|\\s+$"; "")) +
        "\nDocument: " + .document_keyword +
        "\n" + ("-" * 50) + "\n"
    '
}

# Parse command line arguments
while getopts "k:n:d:h" opt; do
    case $opt in
        k)
            KNOWLEDGE_BASE="$OPTARG"
            ;;
        n)
            if [[ "$OPTARG" =~ ^[0-9]+$ ]] && [[ "$OPTARG" -gt 0 ]]; then
                NUM_HITS="$OPTARG"
            else
                echo "Error: Number of hits must be a positive integer" >&2
                exit 1
            fi
            ;;
        d)
            if [[ "$OPTARG" =~ ^[0-3]$ ]]; then
                DEEPENING_LEVEL="$OPTARG"
            else
                echo "Error: Deepening level must be 0-3" >&2
                exit 1
            fi
            ;;
        h)
            usage
            ;;
        \?)
            echo "Error: Invalid option -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Error: Option -$OPTARG requires an argument" >&2
            usage
            ;;
    esac
done

# Shift to get the query argument
shift $((OPTIND-1))

# Get the query from remaining arguments
if [[ $# -eq 0 ]]; then
    echo "Error: Query is required" >&2
    usage
fi

QUERY="$*"

# Validate required parameters
if [[ -z "$KNOWLEDGE_BASE" ]]; then
    echo "Error: Knowledge base (-k) is required" >&2
    usage
fi

# Check dependencies and read configuration
check_dependencies
read_config

# Execute search
echo "Searching knowledge base '$KNOWLEDGE_BASE' for: $QUERY"
echo "Using DSPy query deepening level: $DEEPENING_LEVEL"
echo "Requesting top $NUM_HITS results..."
echo ""

DATASET_ID=$(get_dataset_id "$KNOWLEDGE_BASE")
RESPONSE=$(search_knowledge_base_deepened "\"$DATASET_ID\"" "$QUERY" "$NUM_HITS" "$DEEPENING_LEVEL")
format_results_deepened "$RESPONSE"