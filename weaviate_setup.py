import weaviate

client = weaviate.Client("http://weaviate:8080")

# Example: Create a schema
class_obj = {
    "class": "Article",
    "description": "A collection of articles",
    "properties": [
        {
            "name": "title",
            "dataType": ["text"],
            "description": "Title of the article"
        },
        {
            "name": "content",
            "dataType": ["text"],
            "description": "Content of the article"
        }
    ]
}

client.schema.create_class(class_obj)
print("Schema created.")
