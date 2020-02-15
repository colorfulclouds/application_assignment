import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.lang.*;

public class Main extends JFrame
{
	public class MyPanel extends Panel implements KeyListener
	{
		private int r;

		private Color color;

		public void setColor(Color color) 
		{
			this.color = color;
		}

		public MyPanel(int r, Color color)
		{

			this.r = r;

			this.color = color;

			this.setBackground(Configure.background);
		}

		public void paint(Graphics g)
		{

			g.setColor(color);

			int h = this.getHeight();

			int w = this.getWidth();

			g.fillOval(w/2 - r, h/2 - r, 2 * r, 2 * r);

		}
		
		public void keyTyped(KeyEvent e) 
		{
        // TODO Auto-generated method stub

		}
		public void keyPressed(KeyEvent e) 
		{
			if (e.getKeyCode()==KeyEvent.VK_ESCAPE){
				System.exit(0);
			} 
		}

		public void keyReleased(KeyEvent e) {

		}

	}
	
	// 可以去改变配置文件中的值去操作频率颜色等
	public interface Configure 
	{
		// 频率
		//long FREQ_P_S = 5; //Hz
		
		// 背景色（黑色）
		Color background = Color.BLACK;
		// 球的颜色（白色）
		Color ballColor = Color.WHITE;
		// 球的半径
		int r = 500;
		
		// 小球显示与消失时间（毫秒 必须相等）
		//long visible_hide = (int)(500/FREQ_P_S);

	}
	
    MyPanel jp_center;
	
    public static void main(String[] args) 
	{
		long FREQ_P_S;

		if(args.length == 0)
			FREQ_P_S = 20;
		else
			FREQ_P_S = Integer.valueOf(args[0]);
		
		long visible_hide = (int)(500/FREQ_P_S);
		
        new Main(visible_hide);
    }

    Main(long visible_hide)
	{
        // 新建一个球
        jp_center = new MyPanel(Configure.r,Configure.ballColor);
		// 监听键盘
		this.addKeyListener(jp_center);
        // 设置布局
        this.setLayout(new BorderLayout());
        // 把球加入布局
        this.add(jp_center, BorderLayout.CENTER);
        //去处边框
        this.setUndecorated(true);
        //最大化
        this.setExtendedState(JFrame.MAXIMIZED_BOTH);
        //总在最前面
        this.setAlwaysOnTop(true);
        //不能改变大小
        this.setResizable(false);
        //设置关闭
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        // 可见性
        this.setVisible(true);

        this.setLocationRelativeTo(null);
		
        while(true)
		{
            try
			{
                jp_center.setColor(Configure.ballColor);
                jp_center.repaint();
                Thread.sleep(visible_hide);
				
                jp_center.setColor(Configure.background);
                jp_center.repaint();
                Thread.sleep(visible_hide);
				
            }
			catch(Exception ex)
			{
            }
        }
    }
}
