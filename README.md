# ReflectGPT!

Add the ability for an LLM to think whether the answer it's giving is good enough, and if not to restart, otherwise continue!

Think about what we can do with this:
1. Add new instructions dynamically
2. Link multiple LLMs together, to stop/ error correct/ restart generation
3. Add new pieces of information if an answer isn't good enough, and restart the generation
4. Dynamic insertion based on rules if it's not being satisfied

Eventually, when LLMs are used all over the place, we'll need something like this. This was my proof of concept to test the idea.

![Vroom!](utils/image.png)