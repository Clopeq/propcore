#!/bin/bash



options=("major" "minor" "patch" "Exit")
selected=0

while true; do
    clear
    echo "Current version: v$(uv version --short)"
    echo "Select an option (↑ ↓, Enter):"
    echo

    for i in "${!options[@]}"; do
        if [[ $i -eq $selected ]]; then
            echo -e "  \033[1;32m> ${options[$i]}\033[0m"
        else
            echo "    ${options[$i]}"
        fi
    done

    IFS= read -rsn1 key

    # Handle escape sequences (arrow keys)
    if [[ $key == $'\x1b' ]]; then
        read -rsn2 key
        case "$key" in
            '[A') ((selected--)) ;;
            '[B') ((selected++)) ;;
        esac
    elif [[ $key == "" ]]; then
        break
    fi

    # Wrap selection around
    ((selected < 0)) && selected=$((${#options[@]} - 1))
    ((selected >= ${#options[@]})) && selected=0
done

clear
if [[ "${options[$selected]}" == "Exit" ]]; then
    echo "Exiting without releasing."
    exit 0
fi
uv version -- bump "${options[$selected]}"
V=$(uv version --short)
git add pyproject.toml # or git add .
git commit -m "Release $V"
git tag "v$V"
git push origin HEAD "v$V"
echo "Released v$V"