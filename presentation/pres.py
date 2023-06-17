from manim import *
from manim_slides.slide import Slide
import random
import math

# TeX stuff
tex_template = TexTemplate()
tex_template.add_to_preamble(r"\usepackage[T1, T2A]{fontenc}", prepend=True)
tex_template.add_to_preamble(r"\usepackage[russian, english]{babel}", prepend=True)
tex_template.add_to_preamble(r"\usepackage[utf8]{inputenc}", prepend=True)


# Axes for 2d example
x_max = 10
y_max = 5
example_ax = Axes(
    x_range=[0, x_max, 1],
    x_length=x_max,
    y_range=[0, y_max, 1],
    y_length=y_max,
    tips=False,
)
labels = example_ax.get_axis_labels(
    Tex("размер пятен", tex_template=tex_template).scale(0.5),
    Tex("длинна шипов", tex_template=tex_template).scale(0.5),
)

# Dots for 2d example
x_values = []
y_values = []
random.seed(1)
for _ in range(200):
    x_values.append(round(random.uniform(0.0, x_max), 2))
    y_values.append(round(random.uniform(0.0, y_max), 2))

dots = example_ax.plot_line_graph(
    x_values=x_values, y_values=y_values, vertex_dot_style=dict(fill_color=GRAY)
)["vertex_dots"]


class NeuralNetworkPresentation(Slide):
    def tinywait(self):
        self.wait(0.1)

    def all_fadeout(self):
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def title_slide(self):
        title = VGroup(
            Tex("Нейронная сеть своими руками!", tex_template=tex_template).scale(1),
            Tex("Матюхин Григорий", tex_template=tex_template).scale(0.5),
        ).arrange(DOWN)
        self.play(Write(title))

        self.next_slide()
        self.all_fadeout()

    def a_silly_example(self):
        fruit0 = ImageMobject("assets/fruit0.png").scale(0.7).shift(3 * LEFT)
        fruit1 = ImageMobject("assets/fruit1.png").scale(0.7)
        fruit2 = ImageMobject("assets/fruit2.png").scale(0.7).shift(3 * RIGHT)

        self.play(FadeIn(fruit1))
        self.tinywait()
        self.next_slide()

        self.play(FadeIn(fruit0))
        self.play(FadeIn(fruit2))
        self.tinywait()
        self.next_slide()

        self.all_fadeout()

    def decision_bondary(self):
        self.play(Create(example_ax), Create(labels))
        self.play(Create(dots))
        self.tinywait()
        self.next_slide()

        self.play(
            AnimationGroup(
                *[
                    dot.animate.set_fill(RED)
                    if random.random() > 0.5
                    else dot.animate.set_fill(BLUE)
                    for dot in dots
                ]
            )
        )
        self.next_slide()

        self.play(AnimationGroup(*[dot.animate.set_fill(GRAY) for dot in dots]))

        boundary = lambda x: -0.5 * x + 4

        def is_below_boundary(xy) -> bool:
            (x, y) = xy
            return y < boundary(x)

        self.play(
            AnimationGroup(
                *[
                    dot.animate.set_fill(BLUE)
                    if is_below_boundary(
                        example_ax.point_to_coords([dot.get_x(), dot.get_y(), 0])
                    )
                    else dot.animate.set_fill(RED)
                    for dot in dots
                ]
            )
        )
        self.next_slide()

        graph = example_ax.plot(boundary)
        limited_graph = example_ax.plot(boundary, x_range=(0, 8))
        self.play(Create(limited_graph))

        area_under = example_ax.get_area(limited_graph).set_fill(BLUE_B, opacity=0.5)
        area_over = Difference(
            example_ax.get_area(
                example_ax.plot(function=lambda _: y_max),
                bounded_graph=graph,
                x_range=(0, x_max),
            ),
            example_ax.get_area(
                graph,
                bounded_graph=example_ax.plot(function=lambda _: 0),
                x_range=(8, x_max),
            ),
            color=RED_B,
            fill_opacity=0.5,
            stroke_width=0,
        )

        self.play(FadeIn(area_under))
        self.play(FadeIn(area_over))

        self.next_slide()
        self.all_fadeout()

    def simple_network(self):
        input1 = Dot(4 * LEFT + UP * 0.5, radius=0.1)
        input1_label = MathTex(r"input_1").scale(0.7).next_to(input1, LEFT)
        input2 = Dot(4 * LEFT + DOWN * 0.5, radius=0.1)
        input2_label = MathTex(r"input_2").scale(0.7).next_to(input2, LEFT)

        output1 = Dot(4 * RIGHT, radius=0.1)
        output_label = MathTex(r"output").scale(0.7).next_to(output1, RIGHT)
        output2 = Dot(4 * RIGHT + DOWN * 0.5, radius=0.1)
        output2_label = MathTex(r"output_2").scale(0.7).next_to(output2, RIGHT)

        inputs = VGroup(input1, input2)
        input_labels = VGroup(input1_label, input2_label)

        self.play(Create(inputs), Write(input_labels))
        self.play(Create(output1), Write(output_label))

        rule_label = MathTex(
            r"""output &> 0: {{ \text{безопасный} }} \\
                output &< 0: {{ \text{ядовитый} }}
            """,
            tex_template=tex_template,
        ).to_edge(UP)
        rule_label.set_color_by_tex("безопасный", BLUE)
        rule_label.set_color_by_tex("ядовитый", RED)

        self.play(Write(rule_label))
        self.tinywait()
        self.next_slide()

        output1_label = (
            Tex(r"$output_1$").scale(0.7).next_to(output1, RIGHT).shift(UP * 0.5)
        )
        o1_gr = VGroup(output1, output_label)
        self.play(o1_gr.animate.shift(UP * 0.5))

        rule_label_2 = MathTex(
            r"""output_1 &> output_2: {{ \text{безопасный} }}\\
                output_2 &> output_1: {{ \text{ядовитый} }}
            """,
            tex_template=tex_template,
        ).to_edge(UP)
        rule_label_2.set_color_by_tex("безопасный", BLUE)
        rule_label_2.set_color_by_tex("ядовитый", RED)

        self.play(
            TransformMatchingShapes(output_label, output1_label),
            Create(output2),
            Write(output2_label),
        )
        self.play(TransformMatchingShapes(rule_label, rule_label_2))
        self.next_slide()

        weight_lines = [
            Line(start=input1.get_center(), end=output1.get_center()),
            Line(start=input2.get_center(), end=output1.get_center()),
            Line(start=input1.get_center(), end=output2.get_center()),
            Line(start=input2.get_center(), end=output2.get_center()),
        ]

        weight_labels = [
            MathTex("weight_{1,1}").scale(0.5),
            MathTex("weight_{2,1}").scale(0.5),
            MathTex("weight_{1,2}").scale(0.5),
            MathTex("weight_{2,2}").scale(0.5),
        ]

        l = list(
            zip(
                weight_lines,
                weight_labels,
                [1, 1, 3, 3],
                [
                    (input1, output1),
                    (input2, output1),
                    (input1, output2),
                    (input2, output2),
                ],
            )
        )
        o1_gr, w2 = l[:2], l[2:]

        def create_weights(weights):
            for (line, label, offset, (i, o)) in weights:
                label.next_to(
                    line.get_start() + line.get_unit_vector() * offset, UP, buff=0
                ).rotate(line.get_angle())
                self.play(Create(line))
                self.play(Write(label))
                self.bring_to_front(i, o)
                self.play(
                    label.animate.set_color(GRAY), line.animate.set_color(DARK_GRAY)
                )

        create_weights(o1_gr)
        eq1 = MathTex(
            r"output_1 = input_1 \times weight_{1,1} + input_2 \times weight_{2,1}"
        ).shift(DOWN * 2)
        self.play(Write(eq1))
        self.tinywait()
        self.next_slide()

        create_weights(w2)
        eq2 = MathTex(
            r"output_2 = input_1 \times weight_{1,2} + input_2 \times weight_{2,2}"
        ).next_to(eq1, DOWN)
        self.play(Write(eq2))
        self.tinywait()
        self.next_slide()

        eq1_2 = MathTex(
            r"output_1 = input_1 \times weight_{1,1} + input_2 \times weight_{2,1} + bias_1"
        ).shift(DOWN * 2)
        eq2_2 = MathTex(
            r"output_2 = input_1 \times weight_{1,2} + input_2 \times weight_{2,2} + bias_2"
        ).next_to(eq1_2, DOWN)

        self.play(
            TransformMatchingShapes(eq1, eq1_2), TransformMatchingShapes(eq2, eq2_2)
        )
        self.tinywait()
        self.next_slide()
        self.all_fadeout()

    def neuron_code_example(self):
        logger.warn("Empty slide")
        self.add(Tex("TODO: neuron code example"))
        self.all_fadeout()

    def simple_playground(self):
        self.play(FadeIn(example_ax), FadeIn(labels), FadeIn(dots))
        self.tinywait()
        self.next_slide()

        self.play(
            example_ax.animate.shift(LEFT),
            dots.animate.shift(LEFT),
            labels.animate.shift(LEFT),
        )

        parameters = [
            (0, 0.39, 1, -0.6, 0, 0),
            (0.34, 0.39, 1, -0.6, 0, 0),
            (0.34, 0.39, 0, -0.6, 0, 0),
            (0.34, 0.39, 0, -0.6, -1, 0),
            (0.34, 0.39, 0, -0.6, -1, -0.5),
            (0.34, 0.39, -0.1, -0.6, -1, -0.5),
            (0.34, 0.13, -0.1, -0.6, -1, -0.5),
            (0.34, 0.13, -0.1, -0.52, -1, -0.5),
            (0.34, 0.13, -0.1, -0.52, -1, -0.68),
            (0.34, 0.13, -0.1, -0.52, -1, -1),
            (0.34, 0.13, -0.1, -0.62, -1, -1),
            (0.34, 0.23, -0.1, -0.62, -1, -1),
        ]

        change_flags = [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
        ]

        prev_label = None
        graph = None
        area_under = None
        area_over = None

        for ((w_11, w_12, w_21, w_22, b_1, b_2), change_flag) in zip(
            parameters, change_flags
        ):
            line = lambda i_1: -(i_1 * (w_11 - w_12) + b_1 + b_2) / (w_21 - w_22)

            def line_rev(i_2):
                line_reversed = lambda i_2: -(i_2 * (w_21 - w_22) + b_1 + b_2) / (
                    w_11 - w_12
                )
                try:
                    return line_reversed(i_2)
                except ZeroDivisionError:
                    return x_max

            x_lim = x_max
            if line(0) > 0:
                x_lim = line_rev(0)
                if x_lim < 0:
                    x_lim = x_max
                if x_lim > x_max:
                    x_lim = x_max

            limited_new_graph = example_ax.plot(line, x_range=[0, x_lim])
            new_graph = example_ax.plot(line)

            label = (
                VGroup(
                    Tex("Weights"),
                    MathTex(
                        f"""weight_{{0,1}} &= {{{{ {w_11} }}}}\\\\
                    weight_{{2,1}} &= {{{{ {w_21} }}}}\\\\
                    weight_{{1,2}} &= {{{{ {w_12} }}}}\\\\
                    weight_{{2,2}} &= {{{{ {w_22} }}}}
                    """
                    ),
                    Tex("Biases"),
                    MathTex(
                        f"""bias_{{1}} &= {{{{ {b_1} }}}}\\\\
                    bias_{{2}} &= {{{{ {b_2} }}}}
                    """
                    ),
                )
                .scale(0.5)
                .arrange(DOWN)
                .next_to(example_ax, RIGHT)
            )

            new_area_under = example_ax.get_area(
                limited_new_graph, bounded_graph=example_ax.plot(function=lambda _: 0)
            ).set_fill(BLUE_B, opacity=0.5)

            new_area_over = Difference(
                example_ax.get_area(
                    example_ax.plot(function=lambda _: y_max),
                    bounded_graph=new_graph,
                    x_range=(0, x_max),
                ),
                example_ax.get_area(
                    new_graph,
                    bounded_graph=example_ax.plot(function=lambda _: 0),
                    x_range=(0, x_max),
                ),
                color=RED_B,
                fill_opacity=0.5,
                stroke_width=0,
            )

            self.play(
                Write(label, run_time=2)
                if prev_label is None
                else TransformMatchingShapes(prev_label, label),
                Create(limited_new_graph, run_time=2)
                if graph is None
                else graph.animate.put_start_and_end_on(
                    limited_new_graph.get_start(), limited_new_graph.get_end()
                ),
                Create(new_area_under, run_time=2)
                if area_under is None
                else ReplacementTransform(area_under, new_area_under),
                Create(new_area_over, run_time=2)
                if area_over is None
                else ReplacementTransform(area_over, new_area_over),
            )
            self.wait(1)

            prev_label = label
            if graph is None:
                graph = limited_new_graph
            area_under = new_area_under
            area_over = new_area_over

            if change_flag:
                self.next_slide()

                self.play(
                    FadeOut(graph),
                    FadeOut(area_under),
                    FadeOut(area_over),
                )
                self.next_slide()

                def is_below_boundary(xy) -> bool:
                    (x, y) = xy
                    return ((x - 2) ** 2 + (y + 0.5) ** 2) < 15

                self.play(
                    AnimationGroup(
                        *[
                            dot.animate.set_fill(BLUE)
                            if is_below_boundary(
                                example_ax.point_to_coords(
                                    [dot.get_x(), dot.get_y(), 0]
                                )
                            )
                            else dot.animate.set_fill(RED)
                            for dot in dots
                        ]
                    ),
                )
                self.tinywait()
                self.next_slide()

                self.play(FadeIn(graph), FadeIn(area_under), FadeIn(area_over))
                self.next_slide()

        self.next_slide()
        self.all_fadeout()

    def hidden_layers(self):
        inputs = VGroup(
            Dot(4 * LEFT + UP * 0.5, radius=0.1),
            Dot(4 * LEFT + DOWN * 0.5, radius=0.1),
        )

        outputs = VGroup(
            Dot(4 * RIGHT + UP * 0.5, radius=0.1),
            Dot(4 * RIGHT + DOWN * 0.5, radius=0.1),
        )

        self.play(
            FadeIn(inputs),
            FadeIn(outputs),
        )
        self.play(
            Write(Tex("Ввод", tex_template=tex_template).next_to(inputs, UP)),
            Write(Tex("Вывод", tex_template=tex_template).next_to(outputs, UP)),
        )
        self.tinywait()
        self.next_slide()

        hidden_layer1 = VGroup(
            Dot(UP, radius=0.1),
            Dot(radius=0.1),
            Dot(DOWN, radius=0.1),
        )
        self.play(Create(hidden_layer1))
        hidden_layer_label = Tex("Скрытый слой", tex_template=tex_template).next_to(
            hidden_layer1, UP
        )
        self.play(Write(hidden_layer_label))
        self.tinywait()
        self.next_slide()

        hidden_layer2 = VGroup(
            Dot(UP * 1.5, radius=0.1),
            Dot(UP * 0.5, radius=0.1),
            Dot(DOWN * 0.5, radius=0.1),
            Dot(DOWN * 1.5, radius=0.1),
        )
        hidden_layer_label2 = Tex("Скрытые слои", tex_template=tex_template).next_to(
            hidden_layer2, UP
        )

        self.play(hidden_layer1.animate.shift(RIGHT))
        self.play(TransformMatchingShapes(hidden_layer_label, hidden_layer_label2))
        self.play(Create(hidden_layer2.shift(LEFT)))
        self.tinywait()
        self.next_slide()

        self.play(Uncreate(hidden_layer2))
        self.play(hidden_layer1.animate.shift(LEFT))
        self.play(TransformMatchingShapes(hidden_layer_label2, hidden_layer_label))
        self.tinywait()
        self.next_slide()

        show_explanation = True
        weight_lines1 = []
        for (c, h) in enumerate(hidden_layer1):
            for i in inputs:
                line = Line(start=i.get_center(), end=h.get_center())
                weight_lines1.append(line)
                self.play(Create(line), run_time=0.5)
                self.bring_to_front(i, h)
                self.play(line.animate.set_color(DARK_GRAY), run_time=0.5)

            if show_explanation:
                show_explanation = False
                weight11_label = (
                    MathTex("weight_{1,1,1}", color=GRAY)
                    .scale(0.5)
                    .next_to(
                        weight_lines1[-2].get_start()
                        + weight_lines1[-2].get_unit_vector(),
                        UP,
                        buff=0,
                    )
                    .rotate(weight_lines1[-2].get_angle())
                )
                weight21_label = (
                    MathTex("weight_{2,1,1}", color=GRAY)
                    .scale(0.5)
                    .next_to(
                        weight_lines1[-1].get_start()
                        + weight_lines1[-1].get_unit_vector(),
                        UP,
                        buff=0,
                    )
                    .rotate(weight_lines1[-1].get_angle())
                )
                bias1_label = (
                    MathTex("+ bias_{1,1}", color=GRAY)
                    .scale(0.5)
                    .next_to(weight21_label, UR)
                    .rotate(
                        0.5
                        * (
                            weight_lines1[-2].get_angle()
                            + weight_lines1[-1].get_angle()
                        )
                    )
                )
                self.play(Write(weight11_label))
                self.play(Write(weight21_label))
                self.play(Write(bias1_label))
                self.tinywait()
                self.next_slide()
                self.play(
                    Unwrite(weight11_label),
                    Unwrite(weight21_label),
                    Unwrite(bias1_label),
                )
                self.tinywait()

            wi_label = (
                MathTex(rf"weighted\_input_{{ {c + 1}, 1}}", color=BLUE)
                .scale(0.5)
                .next_to(h, LEFT)
            )
            self.play(Write(wi_label))

        self.next_slide()

        weight_lines2 = []
        for (c, o) in enumerate(outputs):
            for h in hidden_layer1:
                line = Line(start=h.get_center(), end=o.get_center())
                weight_lines2.append(line)
                self.play(Create(line), run_time=0.5)
                self.bring_to_front(o, h)
                self.play(line.animate.set_color(DARK_GRAY), run_time=0.5)

            wi_label = (
                MathTex(rf"weighted\_input_{{ {c + 1}, 2}}", color=GREEN)
                .scale(0.5)
                .next_to(o, LEFT)
            )
            self.play(Write(wi_label))

        self.next_slide()
        self.all_fadeout()

    def activation_function(self):
        label = Tex("Нелинейная функция активации", tex_template=tex_template)
        self.play(Write(label))
        self.tinywait()
        self.next_slide()

        neuron = Circle(radius=1.0, color=BLUE_B, fill_opacity=1)
        input_arrow = Arrow(start=LEFT, end=RIGHT, color=GRAY).next_to(neuron, LEFT)
        input_label = MathTex(r"weighted\_input", color=BLUE).next_to(input_arrow, LEFT)

        output_arrow = Arrow(start=LEFT, end=RIGHT, color=GRAY).next_to(neuron, RIGHT)
        output_label_success = MathTex("1", color=RED).next_to(output_arrow, RIGHT)
        output_label_failure = MathTex("0", color=RED).next_to(output_arrow, RIGHT)
        output_label_activation = MathTex("activation", color=RED).next_to(
            output_arrow, RIGHT
        )

        self.play(label.animate.to_edge(UP))
        self.play(Create(neuron))
        self.play(Write(input_label), Create(input_arrow))
        self.play(Create(output_arrow))
        self.tinywait()
        self.next_slide()

        self.play(Write(output_label_success))
        self.tinywait()
        self.next_slide()

        self.play(TransformMatchingTex(output_label_success, output_label_failure))
        self.tinywait()
        self.next_slide()

        self.play(TransformMatchingTex(output_label_failure, output_label_activation))
        self.tinywait()
        self.next_slide()

        self.play(
            FadeOut(neuron),
            FadeOut(input_arrow),
            FadeOut(output_arrow),
            FadeOut(input_label),
            FadeOut(output_arrow),
            FadeOut(output_label_activation),
        )

        ax = Axes(
            x_range=[-5, 5],
            y_range=[-1, 2],
            tips=False,
            axis_config={"include_numbers": True, "color": GRAY},
        )

        self.play(Create(ax))
        self.tinywait()
        self.next_slide()

        step = lambda x: 0 if x < 0 else 1
        step_graph = ax.plot_line_graph(
            x_values=[-5, 0, 0, 5],
            y_values=[0, 0, 1, 1],
            line_color=RED,
            vertex_dot_style={"fill_opacity": 0},
        )
        step_label = MathTex(
            r"""f(x) =
            \begin{cases}
            0 & x\le 0 \\
            1 & x\geq 0 \\
            \end{cases}
            """
        ).shift(UP + LEFT * 3)
        self.play(Create(step_graph), Write(step_label))
        self.tinywait()
        self.next_slide()

        relu = lambda x: max(0, x)
        relu_line_graph = ax.plot_line_graph(
            x_values=[-5, 0, 2],
            y_values=[0, 0, 2],
            line_color=RED,
            vertex_dot_style={"fill_opacity": 0},
        )
        relu_graph = ax.plot(function=relu, color=RED)
        relu_label = MathTex("f(x) = max(0,x)").shift(UP + LEFT * 3)
        self.play(
            ReplacementTransform(step_graph, relu_line_graph),
            TransformMatchingTex(step_label, relu_label),
        )
        self.tinywait()
        self.next_slide()

        sigmoid = lambda x: 1 / (1 + math.exp(-x))
        sigmoid_graph = ax.plot(function=sigmoid, color=RED)
        sigmoid_label = MathTex(r"\sigma(x) = \frac{1}{1+e^{-x}}").shift(UP + LEFT * 3)
        self.play(
            ReplacementTransform(relu_line_graph, sigmoid_graph),
            TransformMatchingTex(relu_label, sigmoid_label),
        )
        self.tinywait()
        self.next_slide()
        self.all_fadeout()

    def activation_playgroud(self):
        example_ax.shift(RIGHT)
        dots.shift(RIGHT)
        labels.shift(RIGHT)

        self.play(FadeIn(example_ax), FadeIn(labels), FadeIn(dots))
        self.tinywait()
        self.next_slide()

        self.play(
            example_ax.animate.shift(LEFT),
            dots.animate.shift(LEFT),
            labels.animate.shift(LEFT),
        )

        weights = MathTex(
            r"""weight_{1,1,1} &= {{???}}\\
            weight_{2,1,1} &= {{???}}\\
            weight_{2,2,1} &= {{???}}\\
            weight_{2,2,1} &= {{???}}\\
            weight_{2,3,1} &= {{???}}\\
            weight_{2,3,1} &= {{???}}\\
            weight_{1,1,2} &= {{???}}\\
            weight_{2,1,2} &= {{???}}\\
            weight_{3,1,2} &= {{???}}\\
            weight_{1,2,2} &= {{???}}\\
            weight_{2,2,2} &= {{???}}\\
            weight_{3,2,2} &= {{???}}\\
            """
        )
        weights.set_color_by_tex("???", YELLOW)
        biases = MathTex(
            r"""bias_{1,1} &= {{???}}\\
            bias_{2,1} &= {{???}}\\
            bias_{3,1} &= {{???}}\\
            bias_{1,1} &= {{???}}\\
            bias_{2,1} &= {{???}} 
            """
        )
        biases.set_color_by_tex("???", YELLOW)

        label = (
            VGroup(Tex("Weights"), weights, Tex("Biases"), biases)
            .scale(0.5)
            .arrange(DOWN).next_to(example_ax, RIGHT)
        )
        self.play(*[Write(x) for x in label])

        self.next_slide()
        self.all_fadeout()

    def loss_function(self):
        inputs = VGroup(
            Dot(4 * LEFT + UP * 0.5, radius=0.1),
            Dot(4 * LEFT + DOWN * 0.5, radius=0.1),
        )

        hidden_layer = VGroup(
            Dot(UP, radius=0.1),
            Dot(radius=0.1),
            Dot(DOWN, radius=0.1),
        )

        outputs = VGroup(
            Dot(4 * RIGHT + UP * 0.5, radius=0.1),
            Dot(4 * RIGHT + DOWN * 0.5, radius=0.1),
        )

        output_labels = VGroup(
            MathTex("output_1").scale(0.7).next_to(outputs[0], RIGHT),
            MathTex("output_2").scale(0.7).next_to(outputs[1], RIGHT),
        )
        output_safe = VGroup(
            MathTex("output_1 = 1").scale(0.7).next_to(outputs[0], RIGHT),
            MathTex("output_2 = 0").scale(0.7).next_to(outputs[1], RIGHT),
        )
        output_poisonous = VGroup(
            MathTex("output_1 = 0").scale(0.7).next_to(outputs[0], RIGHT),
            MathTex("output_2 = 1").scale(0.7).next_to(outputs[1], RIGHT),
        )

        weights = VGroup(
            # To hidden layer
            Line(
                start=inputs[0].get_center(),
                end=hidden_layer[0].get_center(),
                color=DARK_GRAY,
            ),
            Line(
                start=inputs[0].get_center(),
                end=hidden_layer[1].get_center(),
                color=DARK_GRAY,
            ),
            Line(
                start=inputs[0].get_center(),
                end=hidden_layer[2].get_center(),
                color=DARK_GRAY,
            ),
            Line(
                start=inputs[1].get_center(),
                end=hidden_layer[0].get_center(),
                color=DARK_GRAY,
            ),
            Line(
                start=inputs[1].get_center(),
                end=hidden_layer[1].get_center(),
                color=DARK_GRAY,
            ),
            Line(
                start=inputs[1].get_center(),
                end=hidden_layer[2].get_center(),
                color=DARK_GRAY,
            ),
            # To output layer
            Line(
                start=hidden_layer[0].get_center(),
                end=outputs[0].get_center(),
                color=DARK_GRAY,
            ),
            Line(
                start=hidden_layer[0].get_center(),
                end=outputs[1].get_center(),
                color=DARK_GRAY,
            ),
            Line(
                start=hidden_layer[1].get_center(),
                end=outputs[0].get_center(),
                color=DARK_GRAY,
            ),
            Line(
                start=hidden_layer[1].get_center(),
                end=outputs[1].get_center(),
                color=DARK_GRAY,
            ),
            Line(
                start=hidden_layer[2].get_center(),
                end=outputs[0].get_center(),
                color=DARK_GRAY,
            ),
            Line(
                start=hidden_layer[2].get_center(),
                end=outputs[1].get_center(),
                color=DARK_GRAY,
            ),
        )

        self.play(
            FadeIn(weights),
            FadeIn(inputs),
            FadeIn(hidden_layer),
            FadeIn(outputs),
            FadeIn(output_labels),
        )
        self.tinywait()
        self.next_slide()

        safe = (
            VGroup(
                Dot(radius=0.2, color=BLUE),
                Tex("безопасный", color=BLUE, tex_template=tex_template),
            )
            .arrange(RIGHT)
            .shift(5 * LEFT + UP * 2)
        )

        poisonous = (
            VGroup(
                Dot(radius=0.2, color=RED),
                Tex("ядовитый", color=RED, tex_template=tex_template),
            )
            .arrange(RIGHT)
            .shift(5 * LEFT + DOWN * 2)
        )

        self.play(Create(safe[0]), Write(safe[1]), Create(poisonous[0]), Write(poisonous[1]))
        self.tinywait()
        self.next_slide()

        input_center = inputs.get_center() + LEFT * 0.25
        safe_arrow = CurvedArrow(
            start_point=safe[0].get_center() + DOWN * 0.25,
            end_point=input_center,
            angle=1.6,
            color=BLUE,
        )
        self.play(Create(safe_arrow))
        self.play(
            *[
                TransformMatchingShapes(a, b)
                for (a, b) in zip(output_labels, output_safe)
            ]
        )
        self.next_slide()

        poisonous_arrow = CurvedArrow(
            start_point=poisonous[0].get_center() + UP * 0.25,
            end_point=input_center,
            angle=-1.6,
            color=RED,
        )
        self.play(Uncreate(safe_arrow))
        self.play(Create(poisonous_arrow))
        self.play(
            *[
                TransformMatchingTex(a, b)
                for (a, b) in zip(output_safe, output_poisonous)
            ]
        )

        self.next_slide()
        loss_function_label = MathTex(r"\lambda(x, y)=(x-y)^2").shift(UP * 3)
        self.play(Write(loss_function_label))
        self.tinywait()
        self.next_slide()
        self.all_fadeout()

    def gradient_descent(self):
        ax = Axes(
            x_range=[-4, 4, 1],
            y_range=[-0.5, 4, 1],
            y_length=4,
            tips=False,
        )
        funct = lambda x: 0.2 * x**4 + 0.1 * x**3 - x**2 + 2
        graph = ax.plot(function=funct, x_range=[-2.8, 2.4], color=BLUE_D)
        self.play(Create(ax))
        self.play(Create(graph))
        self.next_slide()

        dot = Dot(radius=0.1)
        line = Line(start=LEFT * 1.5, end=RIGHT * 1.5, color=RED, stroke_width=4)

        def tangent_updater(mob):
            x = ax.point_to_coords(dot.get_center())[0]
            angle = ax.angle_of_tangent(graph=graph, x=x)
            mob.set_angle(angle).move_to(dot.get_center())

        line.add_updater(tangent_updater)

        slope_label = MathTex("")

        def slopre_label_updater(mob):
            x = ax.point_to_coords(dot.get_center())[0]
            slope = round(ax.slope_of_tangent(graph=graph, x=x), 2)
            dir = UP
            if slope > 3:
                dir = UL
            if slope < -2:
                dir = UR
            mob.become(MathTex(f"{slope}")).next_to(dot, dir)

        slope_label.add_updater(slopre_label_updater)

        self.add(line, slope_label)
        self.start_loop()
        self.play(MoveAlongPath(dot, graph, run_time=5))
        graph.reverse_points()
        self.play(MoveAlongPath(dot, graph, run_time=5))
        graph.reverse_points()
        self.end_loop()
        self.remove(line, slope_label, dot)

        extrs = [
            Dot(point=ax.coords_to_point(-1.77972, funct(-1.77972), 0)),
            Dot(point=ax.coords_to_point(0, 2, 0)),
            Dot(point=ax.coords_to_point(1.40472, funct(1.40472), 0)),
        ]

        lines = [
            Line(start=LEFT * 1.5, end=RIGHT * 1.5, color=RED, stroke_width=4).move_to(
                d.get_center()
            )
            for d in extrs
        ]

        for (d, l) in [(Create(d), Create(l)) for (d, l) in zip(extrs, lines)]:
            self.play(l, d)
        self.next_slide()

        for (d, l) in [(Uncreate(d), Uncreate(l)) for (d, l) in zip(extrs, lines)]:
            self.play(l, d)
        self.next_slide()

        def animate_descent(start_x, learn_rate, iters):
            dot = Dot(point=ax.coords_to_point(start_x, funct(start_x), 0), radius=0.1)
            dots = []

            for _ in range(iters):
                x = ax.point_to_coords(dot.get_center())[0]
                slope = ax.slope_of_tangent(graph=graph, x=x)
                new_x = x - slope * learn_rate
                new_dot = Dot(
                    point=ax.coords_to_point(new_x, funct(new_x), 0), radius=0.1
                )
                self.play(Transform(dot, new_dot), dot.animate.set_color(DARK_GRAY))
                dots.append(dot)
                dot = new_dot

            self.tinywait()
            self.next_slide()
            self.play(FadeOut(*dots))

        label = MathTex(r"x_{i+1} = x_i - slope \times learn\_rate").to_edge(UP)
        self.play(FadeIn(label))

        learn_rate = 0.1
        learn_rate_label_1 = MathTex(rf"learn\_rate={learn_rate}").next_to(label, DOWN)
        self.play(FadeIn(learn_rate_label_1))
        animate_descent(-0.5, learn_rate, 12)
        self.next_slide()

        learn_rate = 0.5
        learn_rate_label_2 = MathTex(rf"learn\_rate={learn_rate}").next_to(label, DOWN)

        self.play(
            TransformMatchingTex(
                learn_rate_label_1,
                learn_rate_label_2,
            )
        )
        animate_descent(0.1, learn_rate, 15)
        self.next_slide()

        learn_rate = 0.03
        learn_rate_label_3 = MathTex(rf"learn\_rate={learn_rate}").next_to(label, DOWN)
        self.play(TransformMatchingTex(learn_rate_label_2, learn_rate_label_3))
        animate_descent(2.3, learn_rate, 15)
        self.next_slide()

        self.all_fadeout()

    def calculus(self):
        calculus_label = Tex("Математический анализ", tex_template=tex_template).scale(
            2
        )
        self.play(FadeIn(calculus_label, run_time=0.5))
        self.tinywait()
        self.next_slide()
        self.play(FadeOut(calculus_label))

        input = Dot(4 * LEFT + UP * 3, radius=0.1)
        hidden = Dot(UP * 3, radius=0.1)
        weight_1 = Line(start=input.get_center(), end=hidden.get_center())
        weight_1_label = MathTex("w_1", color=BLUE).scale(0.7).next_to(weight_1, UP)
        output = Dot(4 * RIGHT + UP * 3, radius=0.1)
        weight_2 = Line(start=hidden.get_center(), end=output.get_center())
        weight_2_label = MathTex("w_2", color=BLUE).scale(0.7).next_to(weight_2, UP)

        self.play(Create(input), Create(hidden), Create(output))
        self.play(Create(weight_1), Write(weight_1_label))
        self.play(Create(weight_2), Write(weight_2_label))
        self.tinywait()
        self.next_slide()

        activation_0_label = MathTex("a_0", color=YELLOW).scale(0.7).next_to(input, UP)
        activation_1_label = MathTex("a_1", color=YELLOW).scale(0.7).next_to(hidden, UP)
        activation_2_label = MathTex("a_2", color=YELLOW).scale(0.7).next_to(output, UP)

        formulae_colors = {
            "a_0": YELLOW,
            "z_1": PURPLE_A,
            "w_1": BLUE,
            "b_1": MAROON,
            "A": YELLOW,
            "a_1": YELLOW,
            "z_2": PURPLE_A,
            "w_2": BLUE,
            "b_2": MAROON,
            "a_2": YELLOW,
            "y": BLUE,
            "L": RED,
            r"\lambda": RED,
            ".": BLACK,
        }

        wi_1_formula = (
            MathTex(r"{{z_1}}={{a_0}}{{w_1}}+{{b_1}}")
            .scale(0.7)
            .shift(LEFT * 5 + UP * 2)
        )
        wi_1_formula.set_color_by_tex_to_color_map(formulae_colors)

        activation_1_formula = (
            MathTex("{{a_1}}={{A}}({{z_1}})")
            .scale(0.7)
            .next_to(wi_1_formula, DOWN)
            .align_to(wi_1_formula, LEFT)
        )
        activation_1_formula.set_color_by_tex_to_color_map(formulae_colors)

        wi_2_formula = (
            MathTex(r"{{z_2}}={{a_1}}{{w_2}}+{{b_2}}")
            .scale(0.7)
            .next_to(activation_1_formula, DOWN)
            .align_to(activation_1_formula, LEFT)
        )
        wi_2_formula.set_color_by_tex_to_color_map(formulae_colors)

        activation_2_formula = (
            MathTex("{{a_2}}={{A}}({{z_2}})")
            .scale(0.7)
            .next_to(wi_2_formula, DOWN)
            .align_to(wi_2_formula, LEFT)
        )
        activation_2_formula.set_color_by_tex_to_color_map(formulae_colors)

        self.play(Write(activation_0_label), Write(wi_1_formula))
        self.tinywait()
        self.next_slide()

        self.play(Write(activation_1_label), Write(activation_1_formula))
        self.tinywait()
        self.next_slide()

        self.play(Write(wi_2_formula))
        self.tinywait()
        self.next_slide()

        self.play(Write(activation_2_label), Write(activation_2_formula))
        self.tinywait()
        self.next_slide()

        loss_formula = (
            MathTex(r"{{L}} = {{\lambda}}({{a_2}},{{y}})")
            .scale(0.7)
            .next_to(activation_2_formula, DOWN)
            .align_to(activation_2_formula, LEFT)
        )
        loss_formula.set_color_by_tex_to_color_map(formulae_colors)
        self.play(Write(loss_formula))
        self.tinywait()
        self.next_slide()

        ca_big = CurvedArrow(
            start_point=wi_2_formula[3].get_corner(UL),
            end_point=loss_formula[0].get_corner(UL),
            angle=3,
            stroke_width=2,
            tip_length=0.15,
        )
        self.play(Create(ca_big))
        self.next_slide()
        self.play(FadeOut(ca_big))
        ca = [
            CurvedArrow(
                start_point=wi_2_formula[3].get_corner(UL),
                end_point=wi_2_formula[0].get_corner(UR),
                stroke_width=2,
                tip_length=0.15,
            ),
            CurvedArrow(
                start_point=wi_2_formula[0].get_corner(DL),
                end_point=activation_2_formula[0].get_corner(UL),
                stroke_width=2,
                tip_length=0.15,
            ),
            CurvedArrow(
                start_point=activation_2_formula[0].get_corner(DL),
                end_point=loss_formula[0].get_corner(UL),
                stroke_width=2,
                tip_length=0.15,
            ),
        ]
        for x in ca:
            self.play(Create(x))

        l_on_w2 = (
            MathTex(
                r"{\partial {{L}} \over \partial {{w_2}} .}",
                r"=",
                r"{\partial {{z_2}} \over \partial {{w_2}} .}",
                r"\times",
                r"{\partial {{a_2}} \over \partial {{z_2}} .}",
                r"\times",
                r"{\partial {{L}} \over \partial {{a_2}} .}",
            )
            .scale(0.7)
            .shift(RIGHT * 0.5 + UP * 1.5)
        )
        l_on_w2.set_color_by_tex_to_color_map(formulae_colors)
        self.play(Write(l_on_w2))
        # self.add(index_labels(c_on_w2))
        self.tinywait()
        self.next_slide()

        dl_da = (
            MathTex(
                r"{\partial {{L}} \over \partial {{a_2}} .}",
                r"=",
                r"{{\lambda}}({{a_2}}, {{t}})'",
                r"=",
                r"2({{a_2}} - {{y}})",
            )
            .scale(0.7)
            .next_to(l_on_w2, DOWN)
            .align_to(l_on_w2[21], LEFT)
        )
        dl_da.set_color_by_tex_to_color_map(formulae_colors)
        self.play(Write(dl_da))
        self.tinywait()
        self.next_slide()

        da_dz = (
            MathTex(
                r"{\partial {{a_2}} \over \partial {{z_2}} .}",
                r"=",
                r"{{A}}({{z_2}})'",
                r"=",
                r"\sigma({{z_2}})(1 - \sigma({{z_2}}))",
            )
            .scale(0.7)
            .next_to(l_on_w2, DOWN)
            .align_to(l_on_w2[15], LEFT)
        )
        da_dz.set_color_by_tex_to_color_map(formulae_colors)
        self.play(TransformMatchingTex(dl_da, da_dz))
        self.tinywait()
        self.next_slide()

        dz_dw = (
            MathTex(r"{\partial {{z_2}} \over \partial {{w_2}} .}", r"=", r"{{a_1}}")
            .scale(0.7)
            .next_to(l_on_w2, DOWN)
            .align_to(l_on_w2[9], LEFT)
        )
        dz_dw.set_color_by_tex_to_color_map(formulae_colors)
        self.play(TransformMatchingTex(da_dz, dz_dw))
        self.tinywait()
        self.next_slide()
        self.play(Unwrite(dz_dw))

        self.play(*[FadeOut(a) for a in ca])

        self.next_slide()

        ca_big = CurvedArrow(
            start_point=wi_1_formula[3].get_corner(UL),
            end_point=loss_formula[0].get_corner(UL),
            angle=3,
            stroke_width=2,
            tip_length=0.15,
        )
        self.play(Create(ca_big))
        self.next_slide()
        self.play(FadeOut(ca_big))
        ca = [
            CurvedArrow(
                start_point=wi_1_formula[3].get_corner(UL),
                end_point=wi_1_formula[0].get_corner(UR),
                stroke_width=2,
                tip_length=0.15,
            ),
            CurvedArrow(
                start_point=wi_1_formula[0].get_corner(DL),
                end_point=activation_1_formula[0].get_corner(UL),
                stroke_width=2,
                tip_length=0.15,
            ),
            CurvedArrow(
                start_point=activation_1_formula[0].get_corner(DL),
                end_point=wi_2_formula[0].get_corner(UL),
                stroke_width=2,
                tip_length=0.15,
            ),
            CurvedArrow(
                start_point=wi_2_formula[0].get_corner(DL),
                end_point=activation_2_formula[0].get_corner(UL),
                stroke_width=2,
                tip_length=0.15,
            ),
            CurvedArrow(
                start_point=activation_2_formula[0].get_corner(DL),
                end_point=loss_formula[0].get_corner(UL),
                stroke_width=2,
                tip_length=0.15,
            ),
        ]
        for x in ca:
            self.play(Create(x))

        l_on_w1 = (
            MathTex(
                r"{\partial {{L}} \over \partial {{w_1}} .}",
                r"=",
                r"{\partial {{z_1}} \over \partial {{w_1}} .}",
                r"\times",
                r"{\partial {{a_1}} \over \partial {{z_1}} .}",
                r"\times",
                r"{\partial {{z_2}} \over \partial {{a_1}} .}",
                r"\times",
                r"{\partial {{a_2}} \over \partial {{z_2}} .}",
                r"\times",
                r"{\partial {{L}} \over \partial {{a_2}} .}",
            )
            .scale(0.7)
            .next_to(l_on_w2, DOWN * 2.5)
            .align_to(l_on_w2, LEFT)
        )
        l_on_w1.set_color_by_tex_to_color_map(formulae_colors)
        self.play(Write(l_on_w1))
        self.tinywait()
        self.next_slide()

        dz_da = (
            MathTex(r"{\partial {{z_2}} \over \partial {{a_1}} .}", r"=", r"{{w_2}}")
            .scale(0.7)
            .next_to(l_on_w1, DOWN)
            .align_to(l_on_w1[21], LEFT)
        )
        dz_da.set_color_by_tex_to_color_map(formulae_colors)
        self.play(Write(dz_da))
        self.tinywait()
        self.next_slide()

        self.play(Unwrite(dz_da))

        self.play(
            *[FadeOut(a) for a in ca],
            Unwrite(wi_1_formula),
            Unwrite(activation_1_formula),
            Unwrite(wi_2_formula),
            Unwrite(activation_2_formula),
            Unwrite(loss_formula),
        )
        self.play(
            FadeOut(input),
            FadeOut(activation_0_label),
            FadeOut(hidden),
            FadeOut(activation_1_label),
            FadeOut(output),
            FadeOut(activation_2_label),
            FadeOut(weight_1),
            FadeOut(weight_1_label),
            FadeOut(weight_2),
            FadeOut(weight_2_label),
        )

        self.play(
            l_on_w2.animate.scale(2).move_to(UP * 1.5),
            l_on_w1.animate.scale(2).move_to(DOWN * 1.5),
        )

        bbs1 = Rectangle(height=2, width=4, color=GREEN).move_to(l_on_w2[17])
        bbs2 = Rectangle(height=2, width=4, color=GREEN).move_to(l_on_w1[29])
        bbb = Rectangle(height=2.4, width=8.8, color=BLUE).move_to(l_on_w1[23])

        self.play(Create(bbs1), Create(bbs2))
        self.next_slide()
        self.play(Create(bbb))

        self.next_slide()
        self.all_fadeout()

    def thanks(self):
        title = VGroup(
            Tex("Спасибо за внимание!", tex_template=tex_template).scale(1),
            Tex("https://github.com/gmatiukhin/neural-network-from-scratch", tex_template=tex_template).scale(0.5),
        ).arrange(DOWN)
        self.play(
            Write(title),
            FadeIn(ImageMobject("assets/qr.png").scale(0.3).next_to(title, DOWN))
         )
        self.next_slide()
        self.all_fadeout()

    def construct(self):
        self.title_slide()
        self.a_silly_example()
        self.decision_bondary()
        self.simple_network()
        # self.neuron_code_example()
        self.simple_playground()
        self.hidden_layers()
        self.activation_function()
        self.activation_playgroud()
        self.loss_function()
        self.gradient_descent()
        self.calculus()
        self.thanks()
