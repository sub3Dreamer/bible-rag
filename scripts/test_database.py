from app.translations.verse_repository import VerseRepository


def main():
    repository = VerseRepository()

    # 창세기 1장 1절
    verse = repository.get_verse("krv", "GEN", 1, 1)

    print("=== 창세기 1장 1절 ===")
    print(verse)

    # 창세기 1장 전체
    chapter = repository.get_chapter("krv", "GEN", 1)

    print("\n=== 창세기 1장 ===")
    print(f"전체 절 수: {len(chapter)}")

    for record in chapter[:5]:
        print(f"{record['verse']}절: {record['text']}")


if __name__ == "__main__":
    main()