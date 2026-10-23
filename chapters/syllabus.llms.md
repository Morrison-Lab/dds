# WWU CSCI 481 (Deep Learning) Syllabus

Spring 2027

Published

Last modified: 2026-10-23

> **NOTE:**
>
> This is the Spring 2027 syllabus. Items marked TBD (to be determined) have not been set yet. The syllabus is subject to change; I will announce in class any change made after the first day of class.

## Course Information

Number  
[CSCI 481](https://morrison-lab.github.io/swc/course/CSCI-481.html)

Name  
Deep Learning

Website  
[*Deep Learning* (dds)](https://morrison-lab.github.io/dds/)

Term  
Spring 2027

Credits  
4

Time  
Monday, Tuesday, Wednesday, and Friday, 9:00–9:50am

Place  
CF 420

Office hours  
TBD, in my office (CF 463) unless otherwise noted

Prerequisites  
[DATA 371](https://morrison-lab.github.io/swc/course/DATA-371.html) or [DATA 471](https://morrison-lab.github.io/swc/course/DATA-471.html) (Machine Learning). The machine learning these notes assume is covered in [*Machine Learning for Data Science* (lds)](https://morrison-lab.github.io/lds/), and the mathematics, probability, and statistics behind it in [*Math for Data Science* (mds)](https://morrison-lab.github.io/mds/), [*Probability for Data Science* (pds)](https://morrison-lab.github.io/pds/), and [*Statistics for Data Science* (sds)](https://morrison-lab.github.io/sds/).

CRNs (course reference numbers)  
23277

## Contact Information

Name  
Ezra Morrison

Office  
CF 463

Email  
<ezra.morrison@wwu.edu>, for confidential matters only

All other course correspondence goes through Canvas; see [Communication](#communication).

## Course Description

Covers the fundamentals of deep learning, including important models, such as deep feedforward, recurrent, convolutional and transformer neural networks, and algorithms and considerations for training them. The application of deep learning to domains such as natural language, speech and vision will also be discussed.

## Course Outcomes

On completion of CSCI 481, students will demonstrate:

- An understanding of machine learning fundamentals.
- A thorough understanding of deep feedforward neural networks.
- An understanding of deep learning optimization and regularization.
- An understanding of recurrent and convolutional neural networks.
- The ability to implement deep learning models using modern toolkits.
- The ability to conduct deep learning experiments.

## Textbooks and readings

There are no required textbooks for the course, but several books and notes are recommended:

- [*Deep Learning* (dds)](https://morrison-lab.github.io/dds/), the course notes for CSCI 481.
- *Deep Learning* [\[1\]](#ref-goodfellow2016deep), the main reference text. The book can be viewed at no cost from [the book’s webpage](https://www.deeplearningbook.org/).
- *Probabilistic Machine Learning: An Introduction* [\[2\]](#ref-murphy2022pml), ISBN 9780262046824. The author offers a free PDF on [the book’s webpage](https://probml.github.io/pml-book/).
- *Machine Learning with PyTorch and Scikit-Learn* [\[3\]](#ref-raschka2022mlpytorch), ISBN 9781801819312. Code examples and Jupyter notebooks are available from [the book’s GitHub repository](https://github.com/rasbt/machine-learning-book).
- [*Machine Learning for Data Science* (lds)](https://morrison-lab.github.io/lds/), the notes for the prerequisite machine learning course.

The class will not follow one book closely; use these books and notes as references.

## Materials

I will distribute handouts and homework on Canvas. The lecture notes are on this site.

## Course Agenda

The [course calendar](../chapters/calendar.llms.md) gives the week-by-week agenda, including all deadlines.

## Communication

**Outbound.** I will send official class announcements through Canvas Announcements. Please set Canvas to notify you of them right away, so you do not miss important information.

**Inbound.** All course-related correspondence goes through Canvas. Nearly all questions or comments from you to me and/or the grader should be posted on the official class discussion forum, which is Canvas Discussions. Anything that should not be posted there, such as a grading question or objection, goes in a Canvas message to me and the grader.

The only class-related matters that should not go through Canvas at all are those containing confidential information (e.g., information about your health). Communications containing confidential information should be sent to my WWU email address, which is the one exception to using Canvas.

## Questions and Corrections

If something in the course materials seems wrong, ask. The course materials include the notes, handouts, assignments, and anything I say or show in class, including my slides. I will make mistakes, and so will the other authors of the materials.

Do not assume that you are the confused one when the material might be wrong. [Richard Feynman](https://en.wikipedia.org/wiki/Richard_Feynman) told a room of science teachers that “science is the belief in the ignorance of experts” [\[4\]](#ref-feynman1969science), and in this course I am (supposedly) one of the experts. Ask your question first, in class or outside it, and we can talk through your reasoning together. If you are right, we all learn something and I fix the material. If you are not, we can jointly find out where the confusion started.

## Health Precautions

The health and safety of everyone in the class is a top priority. In short: if you are sick with something that could be communicable, let me know ahead of time and I will try to arrange a remote participation option, so you can stay home and not expose others needlessly. Please contact me if you have questions, concerns or suggestions.

## Software

Homework is written in Python, using [PyTorch](https://pytorch.org/). You will install PyTorch in a Python virtual environment. Expect to implement core methods yourself, first with NumPy and then with PyTorch’s building blocks, before reaching for higher-level libraries.

## Student-Centered Teaching

Student-centered teaching methodologies, which prioritize student learning outcomes over instructor output (“more doing, less listening”), have been shown to be vastly more effective than traditional lectures. As Brian Hutchinson highlights from wearable-sensor studies of electrodermal activity [\[5\]](#ref-poh2010wearable) and his lecture presentation [\[6\]](#ref-hutchinson2024data471), traditional passive lectures produce low sympathetic nervous system arousal and attention—comparable to watching television and sleeping. Active problem-solving, formative assessment, and peer discussion elevate engagement and learning.

Furthermore, cognitive science shows that “the less effort it takes to learn something, the less durable that learning is” [\[7\]](#ref-brown2014makeitstick). This course is intentionally challenging for the right reasons, backed by a growth mindset: talent and mathematical ability can be developed through focused effort, challenges expand your abilities, and mistakes are essential opportunities to grow. Research confirms that an instructor’s growth-mindset beliefs predict student achievement and motivation even after accounting for other faculty characteristics [\[8\]](#ref-canning2019mindset).

Each lecture is built from exercises: I pose a question, you work on it alone or in a small group, and then we discuss solutions together.

### ABCD Cards

On the first day of class, I will give each student an ABCD card, which you will use to vote on answers to questions in class. They work like electronic clickers, but they are free and need no batteries. I use them because they allow both you and me to quickly assess how well you understand the material, so that I can adjust what I emphasize in class and you can adjust what you study out of class to improve learning outcomes. Please bring this card to class each day.

## Grading

Course grades are letter grades, based on:

| Category | Weight |
|:---|:---|
| Five homework assignments (5% each) | 25% |
| Five homework self-grades (1% each) | 5% |
| Three midterm exams (10%, 30%, 30%) | 70% |
| Final exam (optional) | replaces the midterm average, if higher |

I will assign final letter grades based on the weighted average of assignment scores, by the standard scale:

| Cumulative score       | Letter grade |
|:-----------------------|:-------------|
| \\90\\ \le x\\         | A range      |
| \\80\\ \le x \< 90\\\\ | B range      |
| \\70\\ \le x \< 80\\\\ | C range      |
| \\60\\ \le x \< 70\\\\ | D range      |
| \\x \< 60\\\\          | F            |

### Homework

The course has five graded homework assignments, as well as additional ungraded exercises and readings. Each homework assignment is worth 5% of your final grade (25% cumulative). Due dates are on the course calendar.

### Homework Grading

I will post solutions after each homework submission deadline. You are encouraged to self-grade your homework assignments using the posted solutions. Accurately self-grading each homework is 1% of your grade (5% cumulative). The grader and I will spot-check your self-grading, and you are encouraged to ask for help with self-grading during office hours.

Submit your self-graded homework solutions on Canvas one week after I post the solutions. Self-grading should include specific, detailed annotations on each sub-problem, marking:

- where you made mistakes;
- what your mistake was;
- what a correct answer would have been.

In other words, give yourself the feedback that you think an ideal grader would give.

You are also welcome to opt out of self-grading, in whole or in part. If you do not complete your self-grading for an assignment by the corresponding deadline, we will assume that you want us to grade it for you.

All questions and objections to grading must be submitted in writing, as a private Canvas message to both me and the grader (see [Communication](#communication)).

### Exams

The course has three in-class midterm exams, on dates still TBD, which the course calendar will list. There is also an in-person final exam during finals week (TBD). If a midterm’s date changes, I will announce the change at least two weeks before that exam is given.

- I will provide a standardized note sheet for each exam.
- Any calculator allowed for the SAT is allowed for the exams.
- Midterms 1, 2 and 3 are worth 10%, 30% and 30% of the course grade.
- I hold a review session before each exam.

The final exam is entirely optional. If your grade on the final is higher than the average of your three midterm grades, weighted 10 to 30 to 30, I will use your final grade to replace all three of your midterm grades. The final exam can only improve your grade.

## Deadlines and Late Work Policy

Please refer to the course calendar for the most up-to-date deadlines. I accept no late work. If you miss a deadline or exam due to a medical emergency, please contact me as soon as possible to discuss arrangements. If you know in advance that you will be unable to take an exam at the scheduled time, let me know at least one week before the exam, and we may be able to reschedule it for an earlier day and time.

## Submitting Your Work

You will develop and store homework code in Git version control repositories. Specifically, we will use private GitHub repositories under a course GitHub organization (TBD). If you do not yet have a GitHub account, please create one. Be sure to push your final version before the assignment deadline: I will pull the current default branch of the repository at the due time. For an introduction to Git and GitHub, see the [Git and GitHub](https://morrison-lab.github.io/psc/chapters/github.html) chapter of the [*Principles of Scientific Computing* (psc)](https://morrison-lab.github.io/psc/) notes.

## Program Guidelines

The grader and I will grade your programs on three criteria, in this order:

1.  correctness;
2.  clarity;
3.  efficiency.

**Correctness.** A correct program produces the correct output for a given input, and does not produce unintended side effects. Test each component as you introduce it, and try to break your program in testing.

**Clarity.** The easier it is to read your code, the easier it is to find and remove bugs. Design your program thoughtfully and modularly, and replace two blocks of code that do the same thing with one function called from both places.

**Efficiency.** Your programs should be asymptotically efficient. Beyond that, use vectorized tensor operations and take advantage of the optimizations built into the libraries we use.

## Academic Honesty

As a future professional, your integrity is crucial. It is your responsibility to read and be familiar with the university’s academic dishonesty policy, found in Appendix D of the Western Washington University Course Catalog.

- **Homework:** you may collaborate freely with your classmates, and you may use AI tools, such as ChatGPT or GitHub Copilot.
- **Exams:** in-class exams are solo. Work alone, without help from anyone else and without AI tools or other devices; the only aids allowed are the provided note sheet and a calculator, as described under [Exams](#exams).

Cheating on an exam will result in failing the course (grade of F), and I will file documentation of the incident with the university. When in doubt, ask the instructor.

## Writing to Think

When you are stuck in this course, try putting your question into words in writing. Write it in full sentences, with the numbers or code you used and what you have already tried. In an interview on the *5 Minute Mentor* podcast, [Leslie Lamport](https://en.wikipedia.org/wiki/Leslie_Lamport) says:

> As someone said, “Writing is nature’s way of showing you how fuzzy your thinking is.” If you think you understand something, and don’t write down your ideas, you only think you’re thinking.
>
> — Leslie Lamport [\[9\]](#ref-lamport2019think)

## Student Conduct

I am committed to maintaining an inclusive, supportive, and professional environment in all academic settings, including lectures, labs, and course-related online spaces. I expect students to live up to the [Association for Computing Machinery (ACM) Code of Ethics and Professional Conduct](https://www.acm.org/code-of-ethics).

## Disabilities

Reasonable accommodation for persons with documented disabilities should be established within the first week of class and arranged through Disability Resources for Students: telephone 360-650-3083; email <drs@wwu.edu>; and on the web at [Disability Resources](http://www.wwu.edu/disability/).

## Other Policies

Please read [Syllabi@WWU](https://syllabi.wwu.edu/) for additional, important policies.

## References

\[1\]

I. Goodfellow, Y. Bengio, and A. Courville, *Deep learning*. MIT Press, 2016. Available: <https://www.deeplearningbook.org/>

\[2\]

K. P. Murphy, *Probabilistic machine learning: An introduction*. in Adaptive computation and machine learning. MIT Press, 2022.

\[3\]

S. Raschka, Y. (Hayden). Liu, and V. Mirjalili, *Machine learning with PyTorch and Scikit-Learn: Develop machine learning and deep learning models with Python*. Birmingham–Mumbai: Packt Publishing, 2022.

\[4\]

R. P. Feynman, “What is science?” *The Physics Teacher*, vol. 7, no. 6, pp. 313–320, 1969, doi: [10.1119/1.2351388](https://doi.org/10.1119/1.2351388).

\[5\]

M.-Z. Poh, N. C. Swenson, and R. W. Picard, “A wearable sensor for unobtrusive, long-term assessment of electrodermal activity,” *IEEE Transactions on Biomedical Engineering*, vol. 57, no. 5, pp. 1243–1252, 2010, doi: [10.1109/TBME.2009.2038487](https://doi.org/10.1109/TBME.2009.2038487).

\[6\]

B. Hutchinson, “DATA 471/571: Machine learning.” Western Washington University, 2024.

\[7\]

P. C. Brown, H. L. Roediger III, and M. A. McDaniel, *Make it stick: The science of successful learning*. Cambridge, MA: Belknap Press of Harvard University Press, 2014.

\[8\]

E. A. Canning, K. Muenks, D. J. Green, and M. C. Murphy, “STEM faculty who believe ability is fixed have larger racial achievement gaps and inspire less student motivation in their classes,” *Science Advances*, vol. 5, no. 2, p. eaau4734, 2019, doi: [10.1126/sciadv.aau4734](https://doi.org/10.1126/sciadv.aau4734).

\[9\]

L. Lamport, “Think and write, with Leslie Lamport.” Interview by Edaena Salinas, 5 Minute Mentor podcast (Mentors.fm), Aug. 2019. Available: <https://mentors.fm/2019/08/13/think-and-write-with-leslie-lamport/>

Back to top
