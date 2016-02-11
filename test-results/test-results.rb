require 'Tioga/FigureMaker'
require './plot_styles.rb'
require 'Dobjects/Function'

class MyPlots

  include Math
  include Tioga
  include FigureConstants
  include MyPlotStyles

  def t
    @figure_maker
  end

  def initialize
    @figure_maker = FigureMaker.default

    t.tex_preview_preamble += "\n\t\\usepackage{mathtools}\n"
    t.tex_preview_preamble += "\n\t\\usepackage[charter]{mathdesign}\n"

    t.save_dir = 'plots'

    t.def_figure('error-plot') do
      mnras_style
      enter_page
      error_plot
    end

    t.def_figure('audacity-test') do
      mnras_style
      enter_page_short
      audacity_test
    end

  end

  def enter_page
    mnras_style

    t.xlabel_shift = 2.0
    t.ylabel_shift = 1.75

    t.default_frame_left   = 0.12
    t.default_frame_right  = 0.94
    t.default_frame_top    = 0.95
    t.default_frame_bottom = 0.12

    t.default_page_width  = 72 * 3.5

    t.default_page_height = t.default_page_width * \
      (t.default_frame_right - t.default_frame_left) / \
      (t.default_frame_top - t.default_frame_bottom)

    t.default_enter_page_function
  end

  def enter_page_short
    mnras_style

    t.xlabel_shift = 2.0
    t.ylabel_shift = 1.75

    t.default_frame_left   = 0.12
    t.default_frame_right  = 0.94
    t.default_frame_top    = 0.92
    t.default_frame_bottom = 0.18

    t.default_page_width  = 72 * 3.5

    $golden_ratio = 1.61803398875

    t.default_page_height = t.default_page_width * \
      (t.default_frame_right - t.default_frame_left) / \
      (t.default_frame_top - t.default_frame_bottom) / $golden_ratio

    t.default_enter_page_function
  end

  def background_grid
    old_line_width = t.line_width
    t.line_width   = 0.5

    ystart = (1.0).log10
    yend   = (300.0).log10

    n = 12
    yvals = Dvector.new(n+1){|i| ystart + (yend-ystart)*i/n}

    m = -1.5

    yvals.each do |y0|
      x0 = (10.0).log10

      x1 = (1000.0).log10
      y1 = y0 + m*(x1-x0)

      t.show_polyline([x0, x1], [y0, y1], LightGray)
    end

    y0 = yvals[-1]
    x0 = (10.0).log10

    x1 = (200.0).log10
    y1 = y0 + m*(x1-x0)

    t.show_text('x'         => x1,
                'y'         => y1 + 0.1,
                'text'      => '$\epsilon \sim t^{-3/2}$',
                'color'     => LightGray,
                'angle'     => -38,
                'scale'     => 0.8,
                'alignment' => ALIGNED_AT_BASELINE)

    t.line_width = old_line_width
  end

  def plot_file(fname, color, title)
    dur, mean, sigma = Dvector.fancy_read(fname)

    t.show_polyline(dur.safe_log10, sigma.safe_log10, color, title)
  end

  def plot_audacity_data
    # plot measurements of a synthetic Audacity signal
    #
    dur, freq, sigma = Dvector.fancy_read('python-audacity-signal.dat')

    freq = (freq-5.0)/5.0 * (3600*24)

    sigma *= (3600*24)/5

    t.show_polyline(dur.safe_log10, freq.abs.safe_log10,
                    Gray, 'Audacity Data', Line_Type_Dot)

    dur.each_index do |i|
      f = (freq[i]).abs

      sigma = 4.0 * (dur[i]/10.0)**-1.5

      dyp = (f+sigma).safe_log10 - f.safe_log10
      dym = f.safe_log10 - (f-sigma).safe_log10

      t.show_error_bars('x' => (dur[i]).log10,
                        'y' => f.safe_log10,
                        'dy_plus' => dyp,
                        'dy_minus' => dym,
                        'color' => Gray)
    end
  end

  def plot_seiko_data
    # plot measurements of a synthetic Audacity signal
    #
    dur, freq, sigma = Dvector.fancy_read('seiko-convergence.dat')

    # realfreq = freq[-1]

    # dur   = dur[0...-1]
    # freq  = freq[0...-1]
    # sigma = sigma[0...-1]

    realfreq = 6.0

    freq = (freq-realfreq)/realfreq * (3600*24)

    sigma *= (3600*24)/5

    t.show_polyline(dur.safe_log10, freq.abs.safe_log10,
                    Black, 'Seiko Watch', Line_Type_Dot)

    dur.each_index do |i|
      f = (freq[i]).abs

      sigma = 4.0 * (dur[i]/10.0)**-1.5

      dyp = (f+sigma).safe_log10 - f.safe_log10
      dym = f.safe_log10 - (f-sigma).safe_log10

      t.show_error_bars('x' => (dur[i]).log10,
                        'y' => f.safe_log10,
                        'dy_plus' => dyp,
                        'dy_minus' => dym,
                        'color' => Black)
    end
  end

  def error_plot
    t.do_box_labels(nil, 'sample duration (s)', 'accuracy (s/day)')

    h = [['44.1k-6Hz-n0.6.dat',  DodgerBlue,    'SNR = 1.5'],
         ['44.1k-6Hz-n0.3.dat',  FireBrick,     'SNR = 3'],
         ['44.1k-6Hz-n0.1.dat',  MidnightBlue,  'SNR = 10'],
         ['44.1k-6Hz-n0.03.dat', DarkGoldenrod, 'SNR = 33']]

    t.xaxis_log_values = t.yaxis_log_values = true

    t.top_edge_type   = AXIS_LINE_ONLY
    t.right_edge_type = AXIS_LINE_ONLY

    t.legend_scale   = 1.0
    t.legend_text_dy = 1.25

    t.show_plot_with_legend('plot_right_margin'  => 0.0,
                            'legend_left_margin' => 0.05,
                            'legend_top_margin'  => 0.6,
                            'legend_scale' => 0.8) do

      t.show_plot([1.0, 3.0, 1.5, -2.5]) do

        background_grid

        # plot the data
        #
        h.each do |fname, color, title|
          plot_file(fname, color, title)
        end

        #plot_audacity_data
        plot_seiko_data

        # add custom top and right axes
        #
        t.right_edge_type = AXIS_WITH_MAJOR_TICKS_AND_NUMERIC_LABELS
        spec = {
          'from'          => [1.0, 1.5],
          'to'            => [3.0, 1.5],
          'loc'           => TOP,
          'ticks_outside' => false,
          'ticks_inside'  => true,
          'major_ticks'   => [(10.0).log10, (30.0).log10, (60.0).log10,
                              (300.0).log10, (600.0).log10],
          'labels'        => ['10 s', '30 s', '1 min', '5 min', '10 min'],
          'shift'         => -1.25
        }
        t.show_axis(spec)

        sec_per_day = 3600.0*24
        spec = {
          'from'              => [3.0, -2.5],
          'to'                => [3.0, 1.5],
          'loc'               => RIGHT,
          'ticks_outside'     => false,
          'ticks_inside'      => true,
          'shift'             => -1.5,
          'angle'             => 180,
          'major_ticks'       => [(1.0e-4 * sec_per_day).log10,
                                  (1.0e-5 * sec_per_day).log10,
                                  (1.0e-6 * sec_per_day).log10,
                                  (1.0e-7 * sec_per_day).log10],
          'labels'            => ['$10^{-4}$', '$10^{-5}$', '$10^{-6}$',
                                  '$10^{-7}$'],
          'minor_tick_length' => 0.0
        }
        t.show_axis(spec)

      end
    end
  end

  def audacity_test
    dur, err, sigma = Dvector.fancy_read('audacity-5Hz-test.dat')

    sec_per_day = 3600.0 * 24
    err *= sec_per_day
    sigma *= sec_per_day

    t.do_box_labels(nil, 'sample duration (s)', 'accuracy (s/day)')

    t.xaxis_log_values = true
    dur.safe_log10!

    t.top_edge_type   = AXIS_LINE_ONLY

    t.show_plot([(3).log10, 3.0, 5, -5]) do
      old_width = t.line_width
      t.line_width = 0.5
      t.show_polyline([(3).log10, 3.0], [0.0, 0.0])
      t.line_width = old_width

      dur.each_index do |i|
        t.show_error_bars('x' => dur[i],
                          'y' => err[i],
                          'dy' => sigma[i],
                          'color' => DodgerBlue)

        t.show_marker('x' => dur[i],
                      'y' => err[i],
                      'marker' => Bullet,
                      'scale' => 0.3)
      end

      # add custom top and right axes
      #
      t.right_edge_type = AXIS_WITH_MAJOR_TICKS_AND_NUMERIC_LABELS
      spec = {
        'from'          => [(3).log10, 5],
        'to'            => [3.0, 5],
        'loc'           => TOP,
        'ticks_outside' => false,
        'ticks_inside'  => true,
        'major_ticks'   => [(10.0).log10, (30.0).log10, (60.0).log10,
                            (300.0).log10, (600.0).log10],
        'labels'        => ['10 s', '30 s', '1 min', '5 min', '10 min'],
        'shift'         => -1.25
      }
      t.show_axis(spec)

    end
  end

end

MyPlots.new

# Local Variables:
#   compile-command: "tioga test-results.rb -s"
# End:
